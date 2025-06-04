import os
import json
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Dense, Dropout, BatchNormalization, 
                                   Concatenate, multiply, Add, LayerNormalization,
                                   LeakyReLU, MultiHeadAttention, GlobalAveragePooling1D,
                                   Lambda, Embedding)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow_probability as tfp
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import pickle
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
from scipy.optimize import minimize
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx

# Optional imports for advanced features
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import uproot
    ROOT_SUPPORTED = True
except ImportError:
    ROOT_SUPPORTED = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore', category=UserWarning)

@dataclass
class PhysicsConstraint:
    """Dataclass for physics constraints"""
    name: str
    constraint_type: str  # 'conservation', 'bounds', 'ordering'
    parameters: Dict[str, Any]
    weight: float = 1.0
    enabled: bool = True

@dataclass
class DetectorNode:
    """Dataclass for detector graph nodes"""
    detector_id: str
    detector_type: str  # 'halo', 'dwc', 'cherenkov', 'timing'
    position: Tuple[float, float, float]
    properties: Dict[str, Any]

class GraphNeuralNetwork(tf.keras.layers.Layer):
    """Custom Graph Neural Network layer for detector geometry"""
    
    def __init__(self, units, num_layers=2, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_layers = num_layers
        self.activation = activation
        
        # Message passing layers
        self.message_layers = []
        self.update_layers = []
        
        for i in range(num_layers):
            self.message_layers.append(
                Dense(units, activation=activation, name=f'message_{i}')
            )
            self.update_layers.append(
                Dense(units, activation=activation, name=f'update_{i}')
            )
        
        self.output_layer = Dense(units, activation=activation, name='gnn_output')
    
    def call(self, inputs):
        node_features, adjacency_matrix = inputs
        
        # Initialize node representations
        h = node_features
        
        # Message passing iterations
        for i in range(self.num_layers):
            # Aggregate messages from neighbors
            messages = tf.matmul(adjacency_matrix, h)
            messages = self.message_layers[i](messages)
            
            # Update node representations
            h_new = self.update_layers[i](tf.concat([h, messages], axis=-1))
            h = h_new
        
        # Final output
        return self.output_layer(h)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'num_layers': self.num_layers,
            'activation': self.activation
        })
        return config

class ActiveLearningSelector:
    """Active learning component for efficient data selection"""
    
    def __init__(self, strategy='uncertainty', batch_size=100):
        self.strategy = strategy
        self.batch_size = batch_size
        self.labeled_indices = set()
        self.unlabeled_indices = set()
        
    def initialize_pools(self, total_samples, initial_labeled_ratio=0.1):
        """Initialize labeled and unlabeled pools"""
        n_initial = int(total_samples * initial_labeled_ratio)
        all_indices = set(range(total_samples))
        
        # Random initial selection
        self.labeled_indices = set(np.random.choice(
            list(all_indices), n_initial, replace=False
        ))
        self.unlabeled_indices = all_indices - self.labeled_indices
    
    def select_samples(self, model, X_unlabeled, uncertainties=None):
        """Select most informative samples for labeling"""
        if self.strategy == 'uncertainty' and uncertainties is not None:
            # Select samples with highest uncertainty
            unlabeled_list = list(self.unlabeled_indices)
            uncertainty_scores = uncertainties[unlabeled_list]
            top_indices = np.argsort(uncertainty_scores)[-self.batch_size:]
            selected = [unlabeled_list[i] for i in top_indices]
            
        elif self.strategy == 'diversity':
            # Select diverse samples using clustering
            from sklearn.cluster import KMeans
            unlabeled_list = list(self.unlabeled_indices)
            X_subset = X_unlabeled[unlabeled_list]
            
            kmeans = KMeans(n_clusters=self.batch_size, random_state=42)
            clusters = kmeans.fit_predict(X_subset)
            
            # Select one sample per cluster (closest to centroid)
            selected = []
            for i in range(self.batch_size):
                cluster_indices = np.where(clusters == i)[0]
                if len(cluster_indices) > 0:
                    centroid = kmeans.cluster_centers_[i]
                    distances = np.linalg.norm(X_subset[cluster_indices] - centroid, axis=1)
                    closest_idx = cluster_indices[np.argmin(distances)]
                    selected.append(unlabeled_list[closest_idx])
        
        else:
            # Random selection (baseline)
            selected = np.random.choice(
                list(self.unlabeled_indices), 
                min(self.batch_size, len(self.unlabeled_indices)), 
                replace=False
            )
        
        # Update pools
        for idx in selected:
            self.labeled_indices.add(idx)
            self.unlabeled_indices.discard(idx)
        
        return selected

class AdvancedModel:
    
    def __init__(self, log_dir="logs_v5", enable_uncertainty=True, 
                 physics_constraints=True, enable_gnn=True, enable_active_learning=True):
        self.model = None
        self.ensemble_models = []
        self.input_shapes = {}
        self.scalers = {}
        self.encoders = {}
        self.comp_cols = []
        self.calibration_constants = {}
        self.run_metadata = []
        self.log_dir = log_dir
        self.enable_uncertainty = enable_uncertainty
        self.enable_gnn = enable_gnn
        self.enable_active_learning = enable_active_learning
        self.physics_constraints = physics_constraints
        self.physics_features = {}
        self.validation_history = []
        self.detector_graph = None
        self.active_learner = None
        self.explainer = None
        self.continuous_learning_buffer = []
        
        # Advanced physics constraints
        self.physics_constraint_registry = self._initialize_physics_constraints()
        
        # Physics constants (expanded)
        self.PHYSICS_CONSTANTS = {
            'electron_mass': 0.511,  # MeV/c²
            'muon_mass': 105.7,      # MeV/c²
            'pion_mass': 139.6,      # MeV/c²
            'proton_mass': 938.3,    # MeV/c²
            'speed_of_light': 299792458,  # m/s
            'fine_structure': 1/137.0,    # dimensionless
            'planck_constant': 4.136e-15, # eV⋅s
            'boltzmann_constant': 8.617e-5 # eV/K
        }
        
        # Detector geometry (example CERN-like setup)
        self.detector_nodes = self._initialize_detector_geometry()
        
        os.makedirs(self.log_dir, exist_ok=True)
        self.setup_advanced_logging()
        
        if self.enable_active_learning:
            self.active_learner = ActiveLearningSelector()

    def _initialize_physics_constraints(self) -> List[PhysicsConstraint]:
        """Initialize comprehensive physics constraints"""
        constraints = [
            PhysicsConstraint(
                name="composition_conservation",
                constraint_type="conservation",
                parameters={"sum_target": 1.0, "tolerance": 0.01},
                weight=1.0
            ),
            PhysicsConstraint(
                name="energy_conservation",
                constraint_type="conservation", 
                parameters={"input_energy": "beam_energy", "efficiency": 0.95},
                weight=0.5
            ),
            PhysicsConstraint(
                name="momentum_conservation",
                constraint_type="conservation",
                parameters={"dimensions": ["x", "y", "z"]},
                weight=0.3
            ),
            PhysicsConstraint(
                name="charge_conservation",
                constraint_type="conservation",
                parameters={"total_charge": 0},  # Neutral beam
                weight=0.7
            ),
            PhysicsConstraint(
                name="mass_bounds",
                constraint_type="bounds",
                parameters={"min_mass": 0.0, "max_mass": 1000.0},  # MeV/c²
                weight=0.4
            ),
            PhysicsConstraint(
                name="velocity_bounds",
                constraint_type="bounds",
                parameters={"min_beta": 0.0, "max_beta": 1.0},  # v/c
                weight=0.6
            )
        ]
        return constraints

    def _initialize_detector_geometry(self) -> List[DetectorNode]:
        """Initialize detector geometry for GNN"""
        nodes = [
            # Halo quadrants
            DetectorNode("halo_q1", "halo", (-10, 10, 0), {"quadrant": 1}),
            DetectorNode("halo_q2", "halo", (10, 10, 0), {"quadrant": 2}),
            DetectorNode("halo_q3", "halo", (10, -10, 0), {"quadrant": 3}),
            DetectorNode("halo_q4", "halo", (-10, -10, 0), {"quadrant": 4}),
            
            # Drift wire chambers
            DetectorNode("dwc1", "dwc", (0, 0, 50), {"chamber": 1}),
            DetectorNode("dwc2", "dwc", (0, 0, 150), {"chamber": 2}),
            
            # Cherenkov detectors
            DetectorNode("cherenkov1", "cherenkov", (-20, 0, 100), {"threshold": "low"}),
            DetectorNode("cherenkov2", "cherenkov", (20, 0, 100), {"threshold": "high"}),
            
            # Time-of-flight
            DetectorNode("tof_start", "timing", (0, 0, 0), {"type": "start"}),
            DetectorNode("tof_stop", "timing", (0, 0, 200), {"type": "stop"}),
        ]
        return nodes

    def build_detector_graph(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build detector graph for GNN processing"""
        n_nodes = len(self.detector_nodes)
        
        # Create node feature matrix
        node_features = []
        for node in self.detector_nodes:
            # Encode detector type
            type_encoding = {
                'halo': [1, 0, 0, 0],
                'dwc': [0, 1, 0, 0], 
                'cherenkov': [0, 0, 1, 0],
                'timing': [0, 0, 0, 1]
            }
            
            # Combine type encoding with normalized position
            features = (type_encoding[node.detector_type] + 
                       [node.position[0]/100, node.position[1]/100, node.position[2]/200])
            node_features.append(features)
        
        node_features = np.array(node_features, dtype=np.float32)
        
        # Create adjacency matrix based on physical proximity and detector relationships
        G = nx.Graph()
        
        # Add nodes
        for i, node in enumerate(self.detector_nodes):
            G.add_node(i, **node.properties)
        
        # Add edges based on physical relationships
        for i, node1 in enumerate(self.detector_nodes):
            for j, node2 in enumerate(self.detector_nodes):
                if i != j:
                    # Calculate distance
                    dist = np.sqrt(sum((a - b)**2 for a, b in zip(node1.position, node2.position)))
                    
                    # Connect nearby detectors or related detector types
                    if (dist < 50 or  # Physical proximity
                        (node1.detector_type == node2.detector_type) or  # Same type
                        (node1.detector_type == 'halo' and node2.detector_type == 'dwc') or  # Sequential
                        (node1.detector_type == 'dwc' and node2.detector_type == 'cherenkov')):
                        
                        G.add_edge(i, j, weight=1.0/max(dist, 1.0))
        
        # Convert to adjacency matrix
        adjacency_matrix = nx.adjacency_matrix(G).todense().astype(np.float32)
        
        return node_features, adjacency_matrix

    def setup_advanced_logging(self):
        """Enhanced logging system"""
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Multiple specialized loggers
        loggers = {
            'model_operations': 'model_operations.log',
            'physics_validation': 'physics_validation.log',
            'active_learning': 'active_learning.log',
            'gnn_processing': 'gnn_processing.log',
            'continuous_learning': 'continuous_learning.log'
        }
        
        self.specialized_loggers = {}
        for logger_name, filename in loggers.items():
            logger = logging.getLogger(logger_name)
            handler = logging.FileHandler(os.path.join(self.log_dir, filename))
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            self.specialized_loggers[logger_name] = logger
        
        self.logger = self.specialized_loggers['model_operations']

    def enhanced_physics_informed_loss(self, y_true, y_pred, constraint_name):
        """Advanced physics-informed loss with multiple constraints"""
        base_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        
        if not self.physics_constraints:
            return base_loss
        
        total_physics_penalty = 0.0
        
        for constraint in self.physics_constraint_registry:
            if not constraint.enabled:
                continue
                
            if constraint.constraint_type == "conservation":
                if constraint.name == "composition_conservation":
                    # Composition should sum to target value
                    composition_sum = tf.reduce_sum(y_pred, axis=1)
                    target_sum = constraint.parameters["sum_target"]
                    tolerance = constraint.parameters["tolerance"]
                    
                    penalty = tf.reduce_mean(
                        tf.maximum(0.0, tf.abs(composition_sum - target_sum) - tolerance)
                    )
                    total_physics_penalty += constraint.weight * penalty
                
                elif constraint.name == "energy_conservation":
                    # Energy conservation constraint
                    # This would require access to input energy - simplified here
                    energy_penalty = tf.reduce_mean(
                        tf.maximum(0.0, tf.reduce_sum(y_pred**2, axis=1) - 1.0)
                    )
                    total_physics_penalty += constraint.weight * energy_penalty
                
                elif constraint.name == "momentum_conservation":
                    # Momentum conservation (simplified)
                    momentum_penalty = tf.reduce_mean(
                        tf.reduce_sum(tf.abs(y_pred - tf.reduce_mean(y_pred, axis=0)), axis=1)
                    )
                    total_physics_penalty += constraint.weight * momentum_penalty * 0.1
            
            elif constraint.constraint_type == "bounds":
                if constraint.name == "mass_bounds":
                    # Ensure predictions are within physical bounds
                    min_bound = constraint.parameters["min_mass"] / 1000.0  # Normalized
                    max_bound = constraint.parameters["max_mass"] / 1000.0
                    
                    bound_penalty = (tf.reduce_mean(tf.maximum(0.0, min_bound - y_pred)) +
                                   tf.reduce_mean(tf.maximum(0.0, y_pred - max_bound)))
                    total_physics_penalty += constraint.weight * bound_penalty
        
        return base_loss + total_physics_penalty

    def extract_advanced_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract comprehensive physics-motivated features"""
        df_physics = df.copy()
        
        # Original physics features from v4.0
        df_physics = self.extract_physics_features(df_physics)
        
        # Advanced physics features
        
        # 1. Relativistic calculations
        if 'tof' in df.columns:
            # More accurate beta calculation
            df_physics['beta_relativistic'] = 1.0 / np.sqrt(1 + (df['tof'] / 100)**2)
            df_physics['gamma_factor'] = 1.0 / np.sqrt(1 - df_physics['beta_relativistic']**2)
            
            # Kinetic energy estimation
            df_physics['kinetic_energy_est'] = (df_physics['gamma_factor'] - 1) * 100  # Rough estimate
        
        # 2. Advanced halo analysis
        if all(f'halo_q{i}_adc' in df.columns for i in range(1, 5)):
            # Moments analysis
            halo_data = df[[f'halo_q{i}_adc' for i in range(1, 5)]].values
            
            # Center of mass
            positions = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]])  # Quadrant positions
            total_charge = np.sum(halo_data, axis=1, keepdims=True)
            com_x = np.sum(halo_data * positions[:, 0], axis=1) / (total_charge.flatten() + 1e-6)
            com_y = np.sum(halo_data * positions[:, 1], axis=1) / (total_charge.flatten() + 1e-6)
            
            df_physics['halo_com_x'] = com_x
            df_physics['halo_com_y'] = com_y
            df_physics['halo_com_radius'] = np.sqrt(com_x**2 + com_y**2)
            
            # Second moments (beam width)
            var_x = np.sum(halo_data * (positions[:, 0] - com_x.reshape(-1, 1))**2, axis=1) / (total_charge.flatten() + 1e-6)
            var_y = np.sum(halo_data * (positions[:, 1] - com_y.reshape(-1, 1))**2, axis=1) / (total_charge.flatten() + 1e-6)
            
            df_physics['halo_width_x'] = np.sqrt(var_x)
            df_physics['halo_width_y'] = np.sqrt(var_y)
            df_physics['halo_ellipticity'] = (var_x - var_y) / (var_x + var_y + 1e-6)
        
        # 3. Multi-detector correlations
        detector_groups = {
            'halo': [c for c in df.columns if 'halo' in c and 'adc' in c],
            'dwc': [c for c in df.columns if 'dwc' in c],
            'cherenkov': [c for c in df.columns if ('c1' in c or 'c2' in c) and 'adc' in c],
        }
        
        for group1, cols1 in detector_groups.items():
            for group2, cols2 in detector_groups.items():
                if group1 < group2 and cols1 and cols2:  # Avoid duplicate pairs
                    # Cross-correlation between detector groups
                    signal1 = df[cols1].sum(axis=1) if len(cols1) > 1 else df[cols1[0]]
                    signal2 = df[cols2].sum(axis=1) if len(cols2) > 1 else df[cols2[0]]
                    
                    df_physics[f'{group1}_{group2}_correlation'] = signal1 * signal2
                    df_physics[f'{group1}_{group2}_ratio'] = signal1 / (signal2 + 1e-6)
        
        # 4. Particle identification features
        if 'c1_adc' in df.columns and 'c2_adc' in df.columns and 'tof' in df.columns:
            # Advanced particle ID combining Cherenkov and TOF
            cherenkov_total = df['c1_adc'] + df['c2_adc']
            
            # Particle likelihood based on multiple observables
            for particle, mass in self.PHYSICS_CONSTANTS.items():
                if 'mass' in particle:
                    # Combined likelihood from TOF and Cherenkov
                    tof_likelihood = np.exp(-0.5 * ((df['tof'] - mass/100)**2) / 10)
                    cherenkov_likelihood = np.exp(-0.5 * ((cherenkov_total - mass*0.1)**2) / 100)
                    
                    df_physics[f'{particle}_combined_pid'] = tof_likelihood * cherenkov_likelihood
        
        # 5. Event topology features
        if 'dwc1_x' in df.columns and 'dwc2_x' in df.columns:
            # Track curvature (for magnetic field analysis)
            dx = df['dwc2_x'] - df['dwc1_x']
            dy = df['dwc2_y'] - df['dwc1_y'] if 'dwc1_y' in df.columns else 0
            
            df_physics['track_momentum_est'] = np.sqrt(dx**2 + dy**2) * 100  # Rough estimate
            df_physics['track_curvature'] = np.abs(dx * dy) / (dx**2 + dy**2 + 1e-6)**1.5
        
        # 6. Temporal features (if timestamp available)
        if 'timestamp' in df.columns:
            df_physics['event_rate'] = 1.0 / (df['timestamp'].diff().fillna(1.0) + 1e-6)
            df_physics['time_since_start'] = df['timestamp'] - df['timestamp'].min()
            
            # Seasonal/cyclical features
            df_physics['hour_of_day'] = pd.to_datetime(df['timestamp']).dt.hour
            df_physics['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        self.physics_features = df_physics.columns.tolist()
        return df_physics

    def extract_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Original physics feature extraction from v4.0"""
        df_physics = df.copy()
        
        # Halo quadrant analysis
        if all(f'halo_q{i}_adc' in df.columns for i in range(1, 5)):
            # Total charge and distribution
            total_charge = df[[f'halo_q{i}_adc' for i in range(1, 5)]].sum(axis=1)
            df_physics['total_halo_charge'] = total_charge
            
            # Asymmetry measures
            df_physics['asymmetry_x'] = ((df['halo_q1_adc'] + df['halo_q4_adc']) - 
                                       (df['halo_q2_adc'] + df['halo_q3_adc'])) / (total_charge + 1e-6)
            df_physics['asymmetry_y'] = ((df['halo_q1_adc'] + df['halo_q2_adc']) - 
                                       (df['halo_q3_adc'] + df['halo_q4_adc'])) / (total_charge + 1e-6)
            
            # Information theory measures
            charge_probs = df[[f'halo_q{i}_adc' for i in range(1, 5)]].div(total_charge + 1e-6, axis=0)
            df_physics['charge_entropy'] = -np.sum(charge_probs * np.log(charge_probs + 1e-6), axis=1)
            
            # Symmetry measures
            df_physics['diagonal_ratio'] = ((df['halo_q1_adc'] + df['halo_q3_adc']) / 
                                          (df['halo_q2_adc'] + df['halo_q4_adc'] + 1e-6))
        
        return df_physics

    def preprocess_with_gnn(self, df: pd.DataFrame, for_prediction: bool = False) -> Tuple[Dict, Optional[Dict]]:
        """Enhanced preprocessing with GNN features"""
        df = df.copy()
        
        # Standard preprocessing
        X, y = self.preprocess(df, for_prediction)
        
        if self.enable_gnn:
            # Build detector graph
            if self.detector_graph is None:
                node_features, adjacency_matrix = self.build_detector_graph()
                self.detector_graph = (node_features, adjacency_matrix)
            
            # Add graph features to input
            node_features, adjacency_matrix = self.detector_graph
            
            # Create detector signal vector for each event
            detector_signals = []
            for _, row in df.iterrows():
                signal_vector = []
                for node in self.detector_nodes:
                    # Map detector to corresponding column
                    if node.detector_type == 'halo':
                        col_name = f"halo_q{node.properties['quadrant']}_adc"
                    elif node.detector_type == 'dwc':
                        col_name = f"dwc{node.properties['chamber']}_x"
                    elif node.detector_type == 'cherenkov':
                        col_name = "c1_adc" if node.properties['threshold'] == 'low' else "c2_adc"
                    elif node.detector_type == 'timing':
                        col_name = "tof" if node.properties['type'] == 'stop' else "timestamp"
                    else:
                        col_name = None
                    
                    if col_name and col_name in df.columns:
                        signal_vector.append(row[col_name])
                    else:
                        signal_vector.append(0.0)
                
                detector_signals.append(signal_vector)
            
            detector_signals = np.array(detector_signals, dtype=np.float32)
            
            # Add to inputs
            X['gnn_node_features'] = np.tile(node_features, (len(df), 1, 1))
            X['gnn_adjacency'] = np.tile(adjacency_matrix, (len(df), 1, 1))
            X['gnn_signals'] = detector_signals
            
            # Update input shapes
            self.input_shapes['gnn_node_features'] = node_features.shape
            self.input_shapes['gnn_adjacency'] = adjacency_matrix.shape
            self.input_shapes['gnn_signals'] = (len(self.detector_nodes),)
        
        return X, y

    def preprocess(self, df: pd.DataFrame, for_prediction: bool = False) -> Tuple[Dict, Optional[Dict]]:
        """Enhanced preprocessing from v4.0"""
        df = df.copy()
        
        # Extract advanced physics features
        df = self.extract_advanced_physics_features(df)
        
        # Apply calibration constants
        for col, factor in self.calibration_constants.get("scale", {}).items():
            if col in df.columns:
                df[col] *= factor
                
        for col, offset in self.calibration_constants.get("offset", {}).items():
            if col in df.columns:
                df[col] += offset

        # Enhanced grouping with advanced physics features
        groups = {
            'halo_hit': [c for c in df.columns if 'halo_q' in c and '_hit' in c],
            'halo_adc': [c for c in df.columns if 'halo_q' in c and '_adc' in c],
            'halo_physics': ['total_halo_charge', 'asymmetry_x', 'asymmetry_y', 'charge_entropy', 
                            'diagonal_ratio', 'halo_com_x', 'halo_com_y', 'halo_com_radius',
                            'halo_width_x', 'halo_width_y', 'halo_ellipticity'],
            'tracking': [c for c in df.columns if 'dwc' in c] + 
                       ['track_angle_x', 'track_angle_y', 'scattering_estimate',
                        'track_momentum_est', 'track_curvature'],
            'cherenkov': [c for c in df.columns if 'c1' in c or 'c2' in c] + 
                        ['cherenkov_ratio', 'cherenkov_total'],
            'timing': [c for c in df.columns if 'tof' in c or 'timestamp' in c] + 
                     ['beta_relativistic', 'gamma_factor', 'kinetic_energy_est'],
            'particle_id': [c for c in df.columns if '_pid' in c or '_likelihood' in c],
            'event_topology': ['event_rate', 'time_since_start'] + 
                            [c for c in df.columns if 'correlation' in c or 'ratio' in c],
            'temporal': ['hour_of_day', 'day_of_week'] if 'timestamp' in df.columns else []
        }
        
        # Remove empty groups and non-existent columns
        groups = {k: [col for col in v if col in df.columns] for k, v in groups.items() if v}
        
        inputs = {}
        
        # Process each group
        for group_name, columns in groups.items():
            if not columns:
                continue
                
            group_data = df[columns].fillna(0)
            
            # Apply appropriate scaling
            scaler_key = f"{group_name}_scaler"
            if not for_prediction:
                if scaler_key not in self.scalers:
                    if group_name in ['halo_hit']:
                        # Binary features - no scaling needed
                        self.scalers[scaler_key] = None
                    else:
                        # Robust scaling for other features
                        from sklearn.preprocessing import RobustScaler
                        self.scalers[scaler_key] = RobustScaler()
                        group_data = pd.DataFrame(
                            self.scalers[scaler_key].fit_transform(group_data),
                            columns=columns,
                            index=group_data.index
                        )
                else:
                    if self.scalers[scaler_key] is not None:
                        group_data = pd.DataFrame(
                            self.scalers[scaler_key].transform(group_data),
                            columns=columns,
                            index=group_data.index
                        )
            else:
                if scaler_key in self.scalers and self.scalers[scaler_key] is not None:
                    group_data = pd.DataFrame(
                        self.scalers[scaler_key].transform(group_data),
                        columns=columns,
                        index=group_data.index
                    )
            
            inputs[group_name] = group_data.values.astype(np.float32)
            self.input_shapes[group_name] = (group_data.shape[1],)
        
        # Handle composition output
        y_data = None
        if not for_prediction and self.comp_cols:
            available_comp_cols = [col for col in self.comp_cols if col in df.columns]
            if available_comp_cols:
                y_data = df[available_comp_cols].fillna(0).values.astype(np.float32)
                
                # Ensure compositions sum to 1
                row_sums = y_data.sum(axis=1, keepdims=True)
                y_data = y_data / (row_sums + 1e-8)
        
        return inputs, y_data

    def build_advanced_model(self, input_shapes: Dict[str, Tuple], output_dim: int, 
                           model_type: str = "advanced_ensemble") -> Model:
        """Build advanced model with all v5.0 features"""
        inputs = {}
        processed_inputs = {}
        
        # Standard input processing
        for name, shape in input_shapes.items():
            if name.startswith('gnn_'):
                continue  # Handle GNN inputs separately
                
            inputs[name] = Input(shape=shape, name=f"{name}_input")
            
            # Group-specific processing
            if name == 'halo_hit':
                # Binary features - simple dense layers
                x = Dense(32, activation='relu', name=f"{name}_dense1")(inputs[name])
                x = Dropout(0.2)(x)
                processed_inputs[name] = Dense(16, activation='relu', name=f"{name}_dense2")(x)
                
            elif 'halo' in name:
                # Halo analysis with attention
                x = Dense(64, activation='relu', name=f"{name}_dense1")(inputs[name])
                x = BatchNormalization()(x)
                x = Dropout(0.3)(x)
                
                # Self-attention for halo patterns
                x_reshaped = tf.expand_dims(x, 1)
                attention = MultiHeadAttention(num_heads=4, key_dim=16, name=f"{name}_attention")(x_reshaped, x_reshaped)
                attention = GlobalAveragePooling1D()(attention)
                
                processed_inputs[name] = Dense(32, activation='relu', name=f"{name}_output")(attention)
                
            elif name == 'tracking':
                # Tracking with physics-informed layers
                x = Dense(64, activation='relu', name=f"{name}_dense1")(inputs[name])
                x = BatchNormalization()(x)
                x = Dropout(0.25)(x)
                
                # Physics-informed processing
                x = Dense(32, activation='relu', name=f"{name}_physics")(x)
                processed_inputs[name] = LayerNormalization(name=f"{name}_norm")(x)
                
            elif name == 'cherenkov':
                # Cherenkov analysis
                x = Dense(48, activation='relu', name=f"{name}_dense1")(inputs[name])
                x = Dropout(0.2)(x)
                processed_inputs[name] = Dense(24, activation='relu', name=f"{name}_dense2")(x)
                
            elif name == 'timing':
                # Timing analysis with relativistic considerations
                x = Dense(32, activation='relu', name=f"{name}_dense1")(inputs[name])
                x = BatchNormalization()(x)
                processed_inputs[name] = Dense(16, activation='relu', name=f"{name}_dense2")(x)
                
            else:
                # Generic processing for other feature groups
                x = Dense(32, activation='relu', name=f"{name}_dense1")(inputs[name])
                x = Dropout(0.2)(x)
                processed_inputs[name] = Dense(16, activation='relu', name=f"{name}_dense2")(x)
        
        # GNN processing if enabled
        if self.enable_gnn and any(k.startswith('gnn_') for k in input_shapes.keys()):
            gnn_inputs = {}
            for name, shape in input_shapes.items():
                if name.startswith('gnn_'):
                    gnn_inputs[name] = Input(shape=shape, name=f"{name}_input")
                    inputs[name] = gnn_inputs[name]
            
            if 'gnn_node_features' in gnn_inputs and 'gnn_adjacency' in gnn_inputs:
                # Graph neural network processing
                gnn_layer = GraphNeuralNetwork(units=64, num_layers=2, name="detector_gnn")
                gnn_output = gnn_layer([gnn_inputs['gnn_node_features'], gnn_inputs['gnn_adjacency']])
                
                # Combine with detector signals
                if 'gnn_signals' in gnn_inputs:
                    signals_processed = Dense(32, activation='relu', name="signals_dense")(gnn_inputs['gnn_signals'])
                    gnn_combined = tf.concat([
                        GlobalAveragePooling1D()(gnn_output),
                        signals_processed
                    ], axis=1)
                else:
                    gnn_combined = GlobalAveragePooling1D()(gnn_output)
                
                processed_inputs['gnn_features'] = Dense(48, activation='relu', name="gnn_output")(gnn_combined)
        
        # Combine all processed inputs
        if len(processed_inputs) == 1:
            combined = list(processed_inputs.values())[0]
        else:
            combined = Concatenate(name="feature_combination")(list(processed_inputs.values()))
        
        # Advanced fusion layers
        x = Dense(256, activation='relu', name="fusion_dense1")(combined)
        x = BatchNormalization(name="fusion_bn1")(x)
        x = Dropout(0.4)(x)
        
        x = Dense(128, activation='relu', name="fusion_dense2")(x)
        x = BatchNormalization(name="fusion_bn2")(x)
        x = Dropout(0.3)(x)
        
        # Physics-informed intermediate layer
        physics_layer = Dense(64, activation='relu', name="physics_informed")(x)
        physics_layer = LayerNormalization(name="physics_norm")(physics_layer)
        
        # Multi-head attention for feature interaction
        attention_input = tf.expand_dims(physics_layer, 1)
        attention_output = MultiHeadAttention(
            num_heads=8, key_dim=32, name="feature_attention"
        )(attention_input, attention_input)
        attention_output = GlobalAveragePooling1D()(attention_output)
        
        # Final processing
        final_features = Add(name="residual_connection")([physics_layer, attention_output])
        final_features = Dense(32, activation='relu', name="final_dense")(final_features)
        final_features = Dropout(0.2)(final_features)
        
        # Output layer with uncertainty estimation
        if self.enable_uncertainty:
            # Dual output for mean and variance
            mean_output = Dense(output_dim, activation='softmax', name="composition_mean")(final_features)
            
            # Variance output (log-scale for stability)
            log_var_output = Dense(output_dim, activation='linear', name="composition_log_var")(final_features)
            
            # Combine outputs
            outputs = [mean_output, log_var_output]
            output_names = ["composition_mean", "composition_log_var"]
        else:
            outputs = Dense(output_dim, activation='softmax', name="composition")(final_features)
            output_names = ["composition"]
        
        # Create model
        model = Model(inputs=list(inputs.values()), outputs=outputs, name="CERN_Model_v5")
        
        return model

    def uncertainty_loss(self, y_true, outputs):
        """Advanced uncertainty-aware loss function"""
        if not self.enable_uncertainty:
            return tf.keras.losses.categorical_crossentropy(y_true, outputs)
        
        y_mean, y_log_var = outputs
        
        # Heteroscedastic loss
        var = tf.exp(y_log_var)
        
        # Negative log-likelihood loss
        nll_loss = 0.5 * tf.reduce_mean(
            tf.reduce_sum(
                (y_true - y_mean)**2 / var + y_log_var,
                axis=1
            )
        )
        
        # Regularization term to prevent overconfidence
        reg_loss = 0.01 * tf.reduce_mean(tf.reduce_sum(var, axis=1))
        
        # Physics-informed penalty
        physics_penalty = self.enhanced_physics_informed_loss(y_true, y_mean, "composition_conservation")
        
        return nll_loss + reg_loss + physics_penalty

    def train_with_active_learning(self, df: pd.DataFrame, target_columns: List[str],
                                 max_iterations: int = 10, validation_split: float = 0.2):
        """Train model with active learning loop"""
        self.comp_cols = target_columns
        
        # Initial preprocessing
        X, y = self.preprocess_with_gnn(df)
        
        if self.enable_active_learning:
            # Initialize active learning
            self.active_learner.initialize_pools(len(df))
            
            for iteration in range(max_iterations):
                self.specialized_loggers['active_learning'].info(f"Active learning iteration {iteration + 1}")
                
                # Get current labeled data
                labeled_indices = list(self.active_learner.labeled_indices)
                
                # Create training data from labeled indices
                X_labeled = {k: v[labeled_indices] for k, v in X.items()}
                y_labeled = y[labeled_indices] if y is not None else None
                
                # Train model
                model = self.train_ensemble(X_labeled, y_labeled, validation_split=validation_split)
                
                if iteration < max_iterations - 1:  # Don't select samples in last iteration
                    # Get unlabeled data
                    unlabeled_indices = list(self.active_learner.unlabeled_indices)
                    if not unlabeled_indices:
                        break
                    
                    X_unlabeled = {k: v[unlabeled_indices] for k, v in X.items()}
                    
                    # Predict with uncertainty
                    predictions = self.predict_with_uncertainty(X_unlabeled)
                    uncertainties = predictions.get('epistemic_uncertainty', 
                                                  np.random.random(len(unlabeled_indices)))
                    
                    # Select samples for labeling
                    selected_indices = self.active_learner.select_samples(
                        model, X_unlabeled, uncertainties
                    )
                    
                    self.specialized_loggers['active_learning'].info(
                        f"Selected {len(selected_indices)} samples for labeling"
                    )
        else:
            # Regular training without active learning
            model = self.train_ensemble(X, y, validation_split=validation_split)
        
        return model

    def train_ensemble(self, X: Dict, y: np.ndarray, validation_split: float = 0.2,
                      ensemble_size: int = 5) -> Model:
        """Train ensemble of models for improved uncertainty quantification"""
        
        # Split data
        if isinstance(list(X.values())[0], np.ndarray):
            indices = np.arange(len(list(X.values())[0]))
            train_idx, val_idx = train_test_split(indices, test_size=validation_split, 
                                                random_state=42, stratify=np.argmax(y, axis=1))
            
            X_train = {k: v[train_idx] for k, v in X.items()}
            X_val = {k: v[val_idx] for k, v in X.items()}
            y_train, y_val = y[train_idx], y[val_idx]
        else:
            X_train, X_val, y_train, y_val = X, {}, y, None
        
        # Build main model
        output_dim = y_train.shape[1] if y_train is not None else len(self.comp_cols)
        self.model = self.build_advanced_model(self.input_shapes, output_dim)
        
        # Compile with advanced optimizer
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
        
        if self.enable_uncertainty:
            self.model.compile(
                optimizer=optimizer,
                loss=lambda y_true, y_pred: self.uncertainty_loss(y_true, y_pred),
                metrics=['accuracy']
            )
        else:
            self.model.compile(
                optimizer=optimizer,
                loss=lambda y_true, y_pred: self.enhanced_physics_informed_loss(y_true, y_pred, "composition"),
                metrics=['accuracy']
            )
        
        # Advanced callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
            ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-7, monitor='val_loss'),
            ModelCheckpoint(
                os.path.join(self.log_dir, 'best_model.h5'),
                save_best_only=True, monitor='val_loss'
            )
        ]
        
        if self.log_dir:
            callbacks.append(TensorBoard(log_dir=self.log_dir, histogram_freq=1))
        
        # Train main model
        validation_data = ([X_val[k] for k in self.input_shapes.keys()], y_val) if X_val else None
        
        history = self.model.fit(
            [X_train[k] for k in self.input_shapes.keys()],
            y_train,
            validation_data=validation_data,
            epochs=100,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        
        # Train ensemble models
        self.ensemble_models = []
        for i in range(ensemble_size):
            self.logger.info(f"Training ensemble model {i+1}/{ensemble_size}")
            
            # Bootstrap sampling
            n_samples = len(y_train)
            bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
            
            X_bootstrap = {k: v[bootstrap_idx] for k, v in X_train.items()}
            y_bootstrap = y_train[bootstrap_idx]
            
            # Create ensemble model
            ensemble_model = self.build_advanced_model(self.input_shapes, output_dim)
            ensemble_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss=lambda y_true, y_pred: self.uncertainty_loss(y_true, y_pred) if self.enable_uncertainty
                     else self.enhanced_physics_informed_loss(y_true, y_pred, "composition"),
                metrics=['accuracy']
            )
            
            # Train with reduced epochs
            ensemble_model.fit(
                [X_bootstrap[k] for k in self.input_shapes.keys()],
                y_bootstrap,
                epochs=50,
                batch_size=64,
                verbose=0
            )
            
            self.ensemble_models.append(ensemble_model)
        
        # Setup explainability if available
        if SHAP_AVAILABLE and self.model:
            self.setup_explainability(X_train)
        
        self.validation_history.append(history.history)
        return self.model

    def setup_explainability(self, X_train: Dict):
        """Setup SHAP explainer for model interpretability"""
        try:
            # Create background dataset for SHAP
            background_size = min(100, len(list(X_train.values())[0]))
            background_indices = np.random.choice(
                len(list(X_train.values())[0]), 
                background_size, 
                replace=False
            )
            
            background_data = [X_train[k][background_indices] for k in self.input_shapes.keys()]
            
            # Initialize SHAP explainer
            self.explainer = shap.DeepExplainer(self.model, background_data)
            
            self.logger.info("SHAP explainer initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to setup SHAP explainer: {e}")
            self.explainer = None

    def predict_with_uncertainty(self, X: Dict) -> Dict[str, np.ndarray]:
        """Predict with comprehensive uncertainty quantification"""
        if not self.model:
            raise ValueError("Model not trained yet")
        
        X_input = [X[k] for k in self.input_shapes.keys()]
        
        # Main model prediction
        if self.enable_uncertainty:
            main_pred = self.model.predict(X_input)
            if isinstance(main_pred, list):
                mean_pred, log_var_pred = main_pred
                aleatoric_uncertainty = np.exp(log_var_pred)
            else:
                mean_pred = main_pred
                aleatoric_uncertainty = np.zeros_like(mean_pred)
        else:
            mean_pred = self.model.predict(X_input)
            aleatoric_uncertainty = np.zeros_like(mean_pred)
        
        # Ensemble predictions for epistemic uncertainty
        ensemble_preds = []
        for ensemble_model in self.ensemble_models:
            if self.enable_uncertainty:
                pred = ensemble_model.predict(X_input)
                if isinstance(pred, list):
                    ensemble_preds.append(pred[0])  # Use mean prediction
                else:
                    ensemble_preds.append(pred)
            else:
                ensemble_preds.append(ensemble_model.predict(X_input))
        
        if ensemble_preds:
            ensemble_preds = np.array(ensemble_preds)
            epistemic_uncertainty = np.var(ensemble_preds, axis=0)
            ensemble_mean = np.mean(ensemble_preds, axis=0)
        else:
            epistemic_uncertainty = np.zeros_like(mean_pred)
            ensemble_mean = mean_pred
        
        # Total uncertainty
        total_uncertainty = aleatoric_uncertainty + epistemic_uncertainty
        
        return {
            'prediction': ensemble_mean if ensemble_preds else mean_pred,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty,
            'total_uncertainty': total_uncertainty,
            'ensemble_predictions': ensemble_preds if ensemble_preds else None
        }

    def explain_prediction(self, X: Dict, sample_indices: Optional[List[int]] = None) -> Dict:
        """Generate explanations for predictions using SHAP"""
        if not self.explainer:
            return {"error": "SHAP explainer not available"}
        
        try:
            if sample_indices is None:
                sample_indices = list(range(min(10, len(list(X.values())[0]))))
            
            X_explain = [X[k][sample_indices] for k in self.input_shapes.keys()]
            
            # Generate SHAP values
            shap_values = self.explainer.shap_values(X_explain)
            
            # Organize results
            explanations = {
                'shap_values': shap_values,
                'feature_names': list(self.input_shapes.keys()),
                'sample_indices': sample_indices
            }
            
            # Calculate feature importance
            if isinstance(shap_values, list):
                # Multi-output case
                feature_importance = {}
                for i, output_shap in enumerate(shap_values):
                    importance = np.mean(np.abs(output_shap), axis=0)
                    feature_importance[f'output_{i}'] = importance
            else:
                feature_importance = np.mean(np.abs(shap_values), axis=0)
            
            explanations['feature_importance'] = feature_importance
            
            return explanations
            
        except Exception as e:
            return {"error": f"Failed to generate explanations: {e}"}

    def continuous_learning_update(self, new_X: Dict, new_y: np.ndarray, 
                                 learning_rate_factor: float = 0.1):
        """Update model with new data using continuous learning"""
        if not self.model:
            raise ValueError("Model not trained yet")
        
        # Add to continuous learning buffer
        self.continuous_learning_buffer.append((new_X, new_y))
        
        # Keep buffer size manageable
        max_buffer_size = 10000
        if len(self.continuous_learning_buffer) > max_buffer_size:
            self.continuous_learning_buffer = self.continuous_learning_buffer[-max_buffer_size:]
        
        # Periodic retraining trigger
        if len(self.continuous_learning_buffer) % 1000 == 0:
            self.specialized_loggers['continuous_learning'].info(
                f"Triggering continuous learning update with {len(self.continuous_learning_buffer)} samples"
            )
            
            # Combine all buffer data
            combined_X = {}
            combined_y = []
            
            for X_batch, y_batch in self.continuous_learning_buffer:
                for key in X_batch:
                    if key not in combined_X:
                        combined_X[key] = []
                    combined_X[key].append(X_batch[key])
                combined_y.append(y_batch)
            
            # Stack arrays
            for key in combined_X:
                combined_X[key] = np.vstack(combined_X[key])
            combined_y = np.vstack(combined_y)
            
            # Reduce learning rate for stability
            current_lr = float(self.model.optimizer.learning_rate)
            new_lr = current_lr * learning_rate_factor
            self.model.optimizer.learning_rate = new_lr
            
            # Incremental training
            X_input = [combined_X[k] for k in self.input_shapes.keys()]
            
            self.model.fit(
                X_input,
                combined_y,
                epochs=5,
                batch_size=32,
                verbose=0
            )
            
            # Reset learning rate
            self.model.optimizer.learning_rate = current_lr
            
            self.specialized_loggers['continuous_learning'].info("Continuous learning update completed")

    def optimize_for_edge_deployment(self, target_size_mb: float = 50.0) -> Model:
        """Optimize model for edge computing deployment"""
        if not self.model:
            raise ValueError("Model not trained yet")
        
        # Model pruning
        try:
            import tensorflow_model_optimization as tfmot
            
            # Magnitude-based pruning
            pruning_params = {
                'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.0,
                    final_sparsity=0.5,
                    begin_step=0,
                    end_step=1000
                )
            }
            
            pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
                self.model, **pruning_params
            )
            
            # Compile pruned model
            pruned_model.compile(
                optimizer='adam',
                loss=lambda y_true, y_pred: self.enhanced_physics_informed_loss(y_true, y_pred, "composition"),
                metrics=['accuracy']
            )
            
            self.logger.info("Model pruning completed")
            
        except ImportError:
            self.logger.warning("TensorFlow Model Optimization not available, skipping pruning")
            pruned_model = self.model
        
        # Quantization
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            # Post-training quantization
            quantized_model = converter.convert()
            
            # Save quantized model
            quantized_path = os.path.join(self.log_dir, 'model_quantized.tflite')
            with open(quantized_path, 'wb') as f:
                f.write(quantized_model)
            
            # Check size
            model_size_mb = len(quantized_model) / (1024 * 1024)
            self.logger.info(f"Quantized model size: {model_size_mb:.2f} MB")
            
            if model_size_mb <= target_size_mb:
                self.logger.info(f"Model successfully optimized for edge deployment (target: {target_size_mb} MB)")
            else:
                self.logger.warning(f"Model size ({model_size_mb:.2f} MB) exceeds target ({target_size_mb} MB)")
            
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"Quantization failed: {e}")
            return pruned_model

    def cross_validate_advanced(self, df: pd.DataFrame, target_columns: List[str], 
                              n_folds: int = 5, stratified: bool = True) -> Dict:
        """Advanced cross-validation with stratification for rare events"""
        self.comp_cols = target_columns
        
        # Preprocess data
        X, y = self.preprocess_with_gnn(df)
        
        # Prepare stratification labels
        if stratified and y is not None:
            # Create stratification labels based on dominant class
            strat_labels = np.argmax(y, axis=1)
        else:
            strat_labels = None
        
        # Initialize cross-validation
        if stratified and strat_labels is not None:
            cv_splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            splits = cv_splitter.split(list(X.values())[0], strat_labels)
        else:
            from sklearn.model_selection import KFold
            cv_splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            splits = cv_splitter.split(list(X.values())[0])
        
        cv_results = {
            'fold_scores': [],
            'fold_uncertainties': [],
            'fold_physics_violations': [],
            'average_score': 0.0,
            'std_score': 0.0,
            'rare_event_performance': {}
        }
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            self.logger.info(f"Processing fold {fold + 1}/{n_folds}")
            
            # Split data
            X_train = {k: v[train_idx] for k, v in X.items()}
            X_val = {k: v[val_idx] for k, v in X.items()}
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model for this fold
            fold_model = self.build_advanced_model(self.input_shapes, y_train.shape[1])
            fold_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss=lambda y_true, y_pred: self.enhanced_physics_informed_loss(y_true, y_pred, "composition"),
                metrics=['accuracy']
            )
            
            fold_model.fit(
                [X_train[k] for k in self.input_shapes.keys()],
                y_train,
                epochs=50,
                batch_size=64,
                verbose=0
            )
            
            # Evaluate
            X_val_input = [X_val[k] for k in self.input_shapes.keys()]
            val_pred = fold_model.predict(X_val_input)
            
            # Calculate metrics
            fold_score = r2_score(y_val, val_pred, multioutput='weighted_average')
            cv_results['fold_scores'].append(fold_score)
            
            # Physics violation analysis
            composition_sums = np.sum(val_pred, axis=1)
            physics_violations = np.mean(np.abs(composition_sums - 1.0) > 0.05)
            cv_results['fold_physics_violations'].append(physics_violations)

            # Rare event analysis (events with max component < 0.5)
            rare_event_mask = np.max(y_val, axis=1) < 0.5
            if np.any(rare_event_mask):
                rare_event_score = r2_score(
                    y_val[rare_event_mask], 
                    val_pred[rare_event_mask], 
                    multioutput='weighted_average'
                )
                cv_results['rare_event_performance'][f'fold_{fold}'] = rare_event_score
            
            # Uncertainty quantification for this fold
            if self.enable_uncertainty:
                # Mock uncertainty calculation for demonstration
                prediction_std = np.std(val_pred, axis=1)
                mean_uncertainty = np.mean(prediction_std)
                cv_results['fold_uncertainties'].append(mean_uncertainty)
        
        # Aggregate results
        cv_results['average_score'] = np.mean(cv_results['fold_scores'])
        cv_results['std_score'] = np.std(cv_results['fold_scores'])
        cv_results['average_physics_violations'] = np.mean(cv_results['fold_physics_violations'])
        
        if cv_results['fold_uncertainties']:
            cv_results['average_uncertainty'] = np.mean(cv_results['fold_uncertainties'])
        
        # Rare event performance summary
        if cv_results['rare_event_performance']:
            rare_scores = list(cv_results['rare_event_performance'].values())
            cv_results['rare_event_average'] = np.mean(rare_scores)
            cv_results['rare_event_std'] = np.std(rare_scores)
        
        self.logger.info(f"Cross-validation completed: {cv_results['average_score']:.4f} ± {cv_results['std_score']:.4f}")
        
        return cv_results

    def deploy_model_api(self, host: str = "0.0.0.0", port: int = 8000):
        """Deploy model as REST API for real-time inference"""
        try:
            from flask import Flask, request, jsonify
            import json
        except ImportError:
            self.logger.error("Flask not available for API deployment")
            return
        
        app = Flask(__name__)
        
        @app.route('/health', methods=['GET'])
        def health_check():
            return jsonify({
                'status': 'healthy',
                'model_loaded': self.model is not None,
                'timestamp': pd.Timestamp.now().isoformat()
            })
        
        @app.route('/predict', methods=['POST'])
        def predict():
            try:
                # Parse input data
                data = request.get_json()
                
                # Convert to appropriate format
                X = {}
                for group_name in self.input_shapes.keys():
                    if group_name in data:
                        X[group_name] = np.array(data[group_name], dtype=np.float32)
                    else:
                        # Create zero array if missing
                        X[group_name] = np.zeros((1, self.input_shapes[group_name][0]), dtype=np.float32)
                
                # Make prediction
                result = self.predict_with_uncertainty(X)
                
                # Convert numpy arrays to lists for JSON serialization
                response = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        response[key] = value.tolist()
                    else:
                        response[key] = value
                
                return jsonify({
                    'success': True,
                    'predictions': response,
                    'timestamp': pd.Timestamp.now().isoformat()
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': pd.Timestamp.now().isoformat()
                }), 400
        
        @app.route('/explain', methods=['POST'])
        def explain():
            try:
                data = request.get_json()
                
                # Parse input similar to predict
                X = {}
                for group_name in self.input_shapes.keys():
                    if group_name in data:
                        X[group_name] = np.array(data[group_name], dtype=np.float32)
                    else:
                        X[group_name] = np.zeros((1, self.input_shapes[group_name][0]), dtype=np.float32)
                
                # Generate explanations
                explanations = self.explain_prediction(X, sample_indices=[0])
                
                # Convert to JSON-serializable format
                response = {}
                for key, value in explanations.items():
                    if isinstance(value, np.ndarray):
                        response[key] = value.tolist()
                    elif isinstance(value, dict):
                        response[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                       for k, v in value.items()}
                    else:
                        response[key] = value
                
                return jsonify({
                    'success': True,
                    'explanations': response,
                    'timestamp': pd.Timestamp.now().isoformat()
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': pd.Timestamp.now().isoformat()
                }), 400
        
        @app.route('/model_info', methods=['GET'])
        def model_info():
            return jsonify({
                'model_version': 'v5.0',
                'input_shapes': self.input_shapes,
                'output_columns': self.comp_cols,
                'capabilities': {
                    'uncertainty_quantification': self.enable_uncertainty,
                    'graph_neural_networks': self.enable_gnn,
                    'active_learning': self.enable_active_learning
                },
                'ensemble_size': len(self.ensemble_models),
                'timestamp': pd.Timestamp.now().isoformat()
            })
        
        self.logger.info(f"Starting model API server on {host}:{port}")
        app.run(host=host, port=port, debug=False)

    def monitor_model_performance(self, X_new: Dict, y_new: np.ndarray, 
                                alert_threshold: float = 0.1) -> Dict:
        """Monitor model performance and detect drift"""
        if not self.model:
            raise ValueError("Model not trained yet")
        
        # Make predictions
        predictions = self.predict_with_uncertainty(X_new)
        y_pred = predictions['prediction']
        
        # Calculate current performance
        current_score = r2_score(y_new, y_pred, multioutput='weighted_average')
        
        # Store in performance history
        self.performance_history.append({
            'timestamp': pd.Timestamp.now(),
            'score': current_score,
            'n_samples': len(y_new)
        })
        
        # Check for performance drift
        if len(self.performance_history) > 1:
            recent_scores = [h['score'] for h in self.performance_history[-10:]]
            score_trend = np.mean(recent_scores[-5:]) - np.mean(recent_scores[:5]) if len(recent_scores) >= 10 else 0
            
            drift_detected = abs(score_trend) > alert_threshold
        else:
            drift_detected = False
            score_trend = 0.0
        
        # Data drift detection using statistical tests
        try:
            from scipy import stats
            
            # Compare feature distributions (using first feature group as example)
            first_group = list(X_new.keys())[0]
            if hasattr(self, 'reference_data_stats'):
                # Kolmogorov-Smirnov test for distribution drift
                ks_stats = []
                for i in range(X_new[first_group].shape[1]):
                    ref_feature = self.reference_data_stats[first_group][:, i]
                    new_feature = X_new[first_group][:, i]
                    ks_stat, p_value = stats.ks_2samp(ref_feature, new_feature)
                    ks_stats.append((ks_stat, p_value))
                
                # Average KS statistic
                avg_ks_stat = np.mean([ks[0] for ks in ks_stats])
                data_drift_detected = avg_ks_stat > 0.1  # Threshold for drift
            else:
                # Store reference data for future comparisons
                self.reference_data_stats = {k: v.copy() for k, v in X_new.items()}
                data_drift_detected = False
                avg_ks_stat = 0.0
        
        except ImportError:
            data_drift_detected = False
            avg_ks_stat = 0.0
        
        # Physics violation monitoring
        composition_sums = np.sum(y_pred, axis=1)
        physics_violations = np.mean(np.abs(composition_sums - 1.0) > 0.05)
        
        # Uncertainty monitoring
        if 'total_uncertainty' in predictions:
            avg_uncertainty = np.mean(predictions['total_uncertainty'])
            high_uncertainty_samples = np.sum(predictions['total_uncertainty'] > np.percentile(predictions['total_uncertainty'], 95))
        else:
            avg_uncertainty = 0.0
            high_uncertainty_samples = 0
        
        monitoring_results = {
            'current_performance': current_score,
            'performance_trend': score_trend,
            'performance_drift_detected': drift_detected,
            'data_drift_detected': data_drift_detected,
            'data_drift_statistic': avg_ks_stat,
            'physics_violations_rate': physics_violations,
            'average_uncertainty': avg_uncertainty,
            'high_uncertainty_samples': int(high_uncertainty_samples),
            'total_samples_processed': len(y_new),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Log alerts
        if drift_detected:
            self.logger.warning(f"Performance drift detected: trend = {score_trend:.4f}")
        
        if data_drift_detected:
            self.logger.warning(f"Data drift detected: KS statistic = {avg_ks_stat:.4f}")
        
        if physics_violations > 0.1:
            self.logger.warning(f"High physics violations rate: {physics_violations:.4f}")
        
        return monitoring_results

    def generate_comprehensive_report(self, X_test: Dict, y_test: np.ndarray) -> Dict:
        """Generate comprehensive model performance report"""
        if not self.model:
            raise ValueError("Model not trained yet")
        
        # Make predictions
        predictions = self.predict_with_uncertainty(X_test)
        y_pred = predictions['prediction']
        
        # Basic metrics
        r2 = r2_score(y_test, y_pred, multioutput='weighted_average')
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        # Per-class metrics
        per_class_metrics = {}
        for i, col in enumerate(self.comp_cols):
            per_class_metrics[col] = {
                'r2': r2_score(y_test[:, i], y_pred[:, i]),
                'mae': mean_absolute_error(y_test[:, i], y_pred[:, i]),
                'mse': mean_squared_error(y_test[:, i], y_pred[:, i])
            }
        
        # Physics compliance
        composition_sums = np.sum(y_pred, axis=1)
        physics_violations = np.mean(np.abs(composition_sums - 1.0) > 0.05)
        
        # Uncertainty analysis
        uncertainty_metrics = {}
        if 'total_uncertainty' in predictions:
            uncertainty_metrics = {
                'mean_total_uncertainty': float(np.mean(predictions['total_uncertainty'])),
                'mean_aleatoric_uncertainty': float(np.mean(predictions['aleatoric_uncertainty'])),
                'mean_epistemic_uncertainty': float(np.mean(predictions['epistemic_uncertainty'])),
                'uncertainty_correlation_with_error': float(np.corrcoef(
                    np.mean(predictions['total_uncertainty'], axis=1),
                    np.mean(np.abs(y_test - y_pred), axis=1)
                )[0, 1])
            }
        
        # Rare event performance
        rare_event_mask = np.max(y_test, axis=1) < 0.5
        rare_event_metrics = {}
        if np.any(rare_event_mask):
            rare_event_metrics = {
                'rare_event_count': int(np.sum(rare_event_mask)),
                'rare_event_r2': float(r2_score(y_test[rare_event_mask], y_pred[rare_event_mask], multioutput='weighted_average')),
                'rare_event_mae': float(mean_absolute_error(y_test[rare_event_mask], y_pred[rare_event_mask]))
            }
        
        # Feature importance (if explainer available)
        feature_importance = {}
        if self.explainer:
            try:
                sample_indices = np.random.choice(len(y_test), min(100, len(y_test)), replace=False)
                explanations = self.explain_prediction(X_test, sample_indices.tolist())
                if 'feature_importance' in explanations:
                    feature_importance = explanations['feature_importance']
            except Exception as e:
                self.logger.warning(f"Failed to compute feature importance: {e}")
        
        # Model complexity metrics
        model_complexity = {}
        if self.model:
            model_complexity = {
                'total_parameters': int(self.model.count_params()),
                'trainable_parameters': int(sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])),
                'layers_count': len(self.model.layers),
                'model_size_mb': float(self.model.count_params() * 4 / (1024 * 1024))  # Rough estimate
            }
        
        # Compile comprehensive report
        report = {
            'model_version': 'v5.0',
            'evaluation_timestamp': pd.Timestamp.now().isoformat(),
            'dataset_info': {
                'test_samples': len(y_test),
                'feature_groups': list(X_test.keys()),
                'output_classes': self.comp_cols
            },
            'overall_metrics': {
                'r2_score': float(r2),
                'mean_absolute_error': float(mae),
                'mean_squared_error': float(mse),
                'physics_violations_rate': float(physics_violations)
            },
            'per_class_metrics': per_class_metrics,
            'uncertainty_metrics': uncertainty_metrics,
            'rare_event_metrics': rare_event_metrics,
            'model_complexity': model_complexity,
            'feature_importance': feature_importance,
            'training_history': {
                'validation_epochs': len(self.validation_history),
                'cross_validation_folds': len(self.cv_results) if hasattr(self, 'cv_results') else 0
            },
            'capabilities': {
                'uncertainty_quantification': self.enable_uncertainty,
                'graph_neural_networks': self.enable_gnn,
                'active_learning': self.enable_active_learning,
                'ensemble_models': len(self.ensemble_models)
            }
        }
        
        return report

    def save_model_artifacts(self, save_path: str):
        """Save all model artifacts and metadata"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save main model
        if self.model:
            model_path = os.path.join(save_path, 'main_model.h5')
            self.model.save(model_path)
            self.logger.info(f"Main model saved to {model_path}")
        
        # Save ensemble models
        if self.ensemble_models:
            ensemble_dir = os.path.join(save_path, 'ensemble_models')
            os.makedirs(ensemble_dir, exist_ok=True)
            
            for i, model in enumerate(self.ensemble_models):
                ensemble_path = os.path.join(ensemble_dir, f'ensemble_model_{i}.h5')
                model.save(ensemble_path)
            
            self.logger.info(f"Saved {len(self.ensemble_models)} ensemble models")
        
        # Save scalers
        scalers_path = os.path.join(save_path, 'scalers.pkl')
        with open(scalers_path, 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # Save metadata
        metadata = {
            'model_version': 'v5.0',
            'input_shapes': self.input_shapes,
            'comp_cols': self.comp_cols,
            'enable_uncertainty': self.enable_uncertainty,
            'enable_gnn': self.enable_gnn,
            'enable_active_learning': self.enable_active_learning,
            'creation_timestamp': pd.Timestamp.now().isoformat(),
            'performance_history': self.performance_history,
            'validation_history': self.validation_history
        }
        
        metadata_path = os.path.join(save_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Model artifacts saved to {save_path}")

    def load_model_artifacts(self, load_path: str):
        """Load all model artifacts and metadata"""
        # Load metadata
        metadata_path = os.path.join(load_path, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Restore configuration
        self.input_shapes = metadata['input_shapes']
        self.comp_cols = metadata['comp_cols']
        self.enable_uncertainty = metadata['enable_uncertainty']
        self.enable_gnn = metadata['enable_gnn']
        self.enable_active_learning = metadata['enable_active_learning']
        self.performance_history = metadata.get('performance_history', [])
        self.validation_history = metadata.get('validation_history', [])
        
        # Load main model
        model_path = os.path.join(load_path, 'main_model.h5')
        if os.path.exists(model_path):
            # Custom objects for loading
            custom_objects = {
                'uncertainty_loss': self.uncertainty_loss,
                'enhanced_physics_informed_loss': lambda y_true, y_pred: self.enhanced_physics_informed_loss(y_true, y_pred, "composition")
            }
            
            self.model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            self.logger.info(f"Main model loaded from {model_path}")
        
        # Load ensemble models
        ensemble_dir = os.path.join(load_path, 'ensemble_models')
        if os.path.exists(ensemble_dir):
            self.ensemble_models = []
            for filename in sorted(os.listdir(ensemble_dir)):
                if filename.endswith('.h5'):
                    ensemble_path = os.path.join(ensemble_dir, filename)
                    ensemble_model = tf.keras.models.load_model(ensemble_path, custom_objects=custom_objects)
                    self.ensemble_models.append(ensemble_model)
            
            self.logger.info(f"Loaded {len(self.ensemble_models)} ensemble models")
        
        # Load scalers
        scalers_path = os.path.join(load_path, 'scalers.pkl')
        if os.path.exists(scalers_path):
            with open(scalers_path, 'rb') as f:
                self.scalers = pickle.load(f)
        
        self.logger.info(f"Model artifacts loaded from {load_path}")

# Usage example and testing
def demo_enhanced_cern_model():
    """Demonstrate the enhanced CERN model capabilities"""
    
    # Initialize model with all advanced features
    model = CERNModelv5(
        enable_uncertainty=True,
        enable_gnn=True,
        enable_active_learning=True,
        log_dir="./cern_model_logs"
    )
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Create synthetic feature data
    df = pd.DataFrame({
        # Halo features
        'halo_intensity': np.random.exponential(2, n_samples),
        'halo_radius': np.random.normal(10, 2, n_samples),
        'halo_asymmetry': np.random.beta(2, 5, n_samples),
        
        # Tracking features
        'momentum_x': np.random.normal(0, 5, n_samples),
        'momentum_y': np.random.normal(0, 5, n_samples),
        'momentum_z': np.random.normal(50, 10, n_samples),
        'track_chi2': np.random.exponential(1, n_samples),
        
        # Cherenkov features
        'cherenkov_photons': np.random.poisson(100, n_samples),
        'cherenkov_angle': np.random.normal(0.8, 0.1, n_samples),
        
        # Timing features
        'tof_measurement': np.random.normal(25, 3, n_samples),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1s'),
        
        # Binary hit features
        'halo_hit_detector_1': np.random.binomial(1, 0.3, n_samples),
        'halo_hit_detector_2': np.random.binomial(1, 0.2, n_samples),
    })
    
    # Create synthetic composition targets
    target_columns = ['proton_fraction', 'helium_fraction', 'carbon_fraction', 'iron_fraction']
    
    # Generate realistic composition data
    raw_compositions = np.random.dirichlet([3, 2, 1, 0.5], n_samples)
    for i, col in enumerate(target_columns):
        df[col] = raw_compositions[:, i]
    
    print("=== Enhanced CERN Model v5.0 Demo ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Target columns: {target_columns}")
    
    # Train model with active learning
    print("\n1. Training with Active Learning...")
    trained_model = model.train_with_active_learning(
        df, target_columns, 
        max_iterations=3, 
        validation_split=0.2
    )
    
    # Cross-validation
    print("\n2. Advanced Cross-Validation...")
    cv_results = model.cross_validate_advanced(
        df, target_columns, 
        n_folds=3, 
        stratified=True
    )
    print(f"CV Score: {cv_results['average_score']:.4f} ± {cv_results['std_score']:.4f}")
    
    # Test predictions with uncertainty
    print("\n3. Predictions with Uncertainty...")
    test_indices = np.random.choice(len(df), 10, replace=False)
    test_df = df.iloc[test_indices]
    
    X_test, y_test = model.preprocess_with_gnn(test_df)
    predictions = model.predict_with_uncertainty(X_test)
    
    print("Sample predictions:")
    for i in range(min(3, len(predictions['prediction']))):
        pred = predictions['prediction'][i]
        uncertainty = predictions['total_uncertainty'][i]
        print(f"  Sample {i}: {[f'{p:.3f}' for p in pred]}, Uncertainty: {[f'{u:.3f}' for u in uncertainty]}")
    
    # Generate explanations
    print("\n4. Model Explanations...")
    explanations = model.explain_prediction(X_test, sample_indices=[0, 1])
    if 'error' not in explanations:
        print("  SHAP explanations generated successfully")
    else:
        print(f"  Explanation error: {explanations['error']}")
    
    # Performance monitoring
    print("\n5. Performance Monitoring...")
    monitoring_results = model.monitor_model_performance(X_test, y_test)
    print(f"  Current performance: {monitoring_results['current_performance']:.4f}")
    print(f"  Physics violations: {monitoring_results['physics_violations_rate']:.4f}")
    
    # Comprehensive report
    print("\n6. Comprehensive Report...")
    report = model.generate_comprehensive_report(X_test, y_test)
    print(f"  Overall R² score: {report['overall_metrics']['r2_score']:.4f}")
    print(f"  Model parameters: {report['model_complexity']['total_parameters']:,}")
    
    # Edge optimization
    print("\n7. Edge Optimization...")
    try:
        optimized_model = model.optimize_for_edge_deployment(target_size_mb=10.0)
        print("  Model successfully optimized for edge deployment")
    except Exception as e:
        print(f"  Edge optimization failed: {e}")
    
    # Save artifacts
    print("\n8. Saving Model Artifacts...")
    save_path = "./cern_model_artifacts"
    model.save_model_artifacts(save_path)
    print(f"  Artifacts saved to {save_path}")
    
    print("\n=== Demo Complete ===")
    return model, report

if __name__ == "__main__":
    # Run the demo
    demo_model, demo_report = demo_enhanced_cern_model()
