# Project Aura: Beam Analysis Using Halo Data and Machine Learning

This repository presents **Project Aura**, a scientific investigation combining **accelerator physics** and **machine learning**.  
It documents the theoretical background, experimental planning, data simulation, modeling, and results interpretation associated with beam halo analysis.  

## Contents of the Repository

- [1. Introduction](#introduction)
- [2. Scientific Background & Methodology](#scientific-background--methodology)
- [3. Data Generator](#data-generator)
- [4. Model](#model)
- [5. Simulation Results](#simulation-results)
- [6. Contingency Plan](#contingency-plan)
- [7. Final Note](#final-note)
---

# Introduction

---

## 1.1 Intro

In accelerator physics, beam halos are often regarded as unwanted side effects — low-density peripheral particles that deviate from the dense beam core, potentially damaging equipment or polluting measurements. However, these halos are not meaningless. Their shape and intensity often encode subtle information about the beam's composition, momentum, stability, and its interaction with materials along the beamline.

---

## 1.2 Research Focus

Our primary research question:

> **Can artificial intelligence and machine learning techniques extract useful diagnostic information from beam halos?**

In simple terms, we seek to study beam halos — typically considered noise — and explore whether machine learning models can reveal hidden patterns related to beam stability, composition, and structure.  

This interdisciplinary approach connects modern accelerator science with the transformative power of AI.

---

# Scientific Background & Methodology

---

## 2.1 Scientific Background

### 2.1.1 Beam Halo Formation

Beam halos primarily originate from **Multiple Coulomb Scattering (MCS)** where charged particles interact electromagnetically with the nuclei and electrons of materials like thin foils or collimators. MCS causes particles to deviate from their original path, forming a diffuse halo. The root-mean-square (RMS) scattering angle, `$ \theta_0 $`, is given by the Highland formula:

When a beam hits thin foils or internal collimators, some particles are deflected at small angles, forming a scattered formation leading to halo creation.
The root-mean-square (RMS) scattering angle, `$ \theta_0 $`, is approximately given by:

$$
\theta_0 \approx \frac{13.6 \text{ MeV}}{\beta pc} \cdot z \cdot \sqrt{\frac{x}{X_0}} \left( 1 + 0.038 \ln{\left(\frac{x}{X_0}\right)} \right) \quad (\text{Eq. 1})
$$

Where:
- `$ \beta = \frac{v}{c} $` is the particle’s velocity normalized to the speed of light
- `$ p $` is the momentum (MeV/c)
- `$ z $` is the particle’s charge
- `$ x $` is the thickness of the material
- `$ X_0 $` is the radiation length of the material

This equation shows that the scattering angle depends on the particle type (via `$ \beta, p, z $`) and material properties (`$ x, X_0 $`), leading to distinct halo patterns for different beams and foils.

---

### 2.1.2 Role of Halo Detectors in Beam Diagnostics


Unlike typical beamline setups that filter out halo particles, our experiment intentionally studies them to understand the beam’s properties.

In this project, four scintillation detectors forming a hollow square pattern downstream of a target region — creating a **Halo Detector (HD) array**, placed downstream of a scattering region (an Al/C/Cu foil to ensure halo creation). 

Each panel measures the intensity of scattered particles using an Analogue-to-Digital Converter (ADC), which converts light signals from particle hits into digital values.

By comparing ADC values across the four panels, we can calculate the halo’s shape and asymmetry, revealing information about the beam’s composition and energy. 
 
This approach is like interpreting a shadow: although the halo detector provides coarse data compared to high-resolution trackers, its patterns are rich enough to diagnose beam conditions — while the resolution may relatively be low, shadows can still reveal properties of the object casting them.

---

## 2.2 Methodology

### 2.2.1 Research Question

> **Can meaningful insights be extracted from beam halo structures using machine learning?**

Specifically, can we:
- Predict beam composition and energy using only halo features?
- Detect or anticipate beam instabilities from halo time evolution?
- Classify halo shape patterns and uncover hidden structures?

---

### 2.2.2 Experimental Tasks Breakdown

| Task | Goal | Method |
|:----:|:----|:------|
| **Task 1** | Beam Composition Classification | Use halo quadrant asymmetries + PID detectors (Cherenkov, ToF, Calorimeter) to classify particle types. |
| **Task 2** | Beam Energy Regression | Correlate energy with angular spread and calorimeter response. |
| **Task 3** | Beam Stability Analysis | Track time-series variance in halo signals across spills to detect instability. |
| **Task 4** | Halo Shape Classification | Cluster events based on quadrant signal ratios to categorize shape profiles (symmetric, skewed, diffuse). |

---

### 2.2.3 Mathematical Tools Employed

- **Momentum-Energy Relation** (for mass estimation via ToF and Cherenkov thresholds):

\[
E^2 = (pc)^2 + (mc^2)^2 \quad (\text{Eq. 2})
\]

Where \( E \) is the total energy, \( p \) is momentum, \( m \) is rest mass, and \( c \) is the speed of light.

- **Cherenkov Threshold Condition**:

\[
\beta > \frac{1}{n} \quad (\text{Eq. 3})
\]

Where \( \beta = \frac{v}{c} \) is the particle’s velocity relative to light, and \( n \) is the refractive index of the Cherenkov detector’s gas. Particles faster than this threshold emit light, identifying their type.

- **Halo Asymmetry Metrics** — used to capture geometric deviation:

\[
A_x = \frac{(Q_1 + Q_4) - (Q_2 + Q_3)}{Q_1 + Q_2 + Q_3 + Q_4} \quad (\text{Eq. 4})
\]

\[
A_y = \frac{(Q_1 + Q_2) - (Q_3 + Q_4)}{Q_1 + Q_2 + Q_3 + Q_4} \quad (\text{Eq. 5})
\]

Where \( Q_1, Q_2, Q_3, Q_4 \) are the ADC values from the four halo detector panels, representing particle hit intensities. \( A_x \) and \( A_y \) measure left-right and top-bottom asymmetries.

- **Time-of-Flight (ToF) Mass Estimation** (from S1 and S2 timing):

\[
m = \frac{p}{c} \sqrt{\left(\frac{c \Delta t}{L}\right)^2 - 1} \quad (\text{Eq. 6})
\]

Where \( \Delta t \) is time-of-flight, and \( L \) is the distance between S1 and S2 (in cm).

---

### 2.2.4 Detector Integration Strategy

| Detector | Role | Method |
|:--------|:-----|:------|
| **S1 & S2 (Scintillators)** | Start/stop for ToF | Fast timing (~200 ps) |
| **DWC1–4 (Delay Wire Chambers)** | Pre/post tracking | Linear track fitting, angular deviation measurement |
| **Cherenkov Detectors (C1, C2)** | PID via velocity thresholding | Tunable pressure settings |
| **CALO (Lead Glass Calorimeter)** | Energy measurement | Used in energy regression and PID cross-checks |
| **Halo Detector (Custom)** | Coarse angular halo information | 4-quadrant ADC vector input to AI model |

---

# Data Generator

---

## 3.1 Data generator

In order to ensure feasibility of the proposed experiment, a synthetic data generator was created to simulate the expected detector readings from a physically logical standpoint.

This approach allows us to:

- Create labelled datasets that mimic expected detector readings.
- Introduce controlled noise and beam variations to test model robustness.
- Ensure that our ML models can generalize to real-world beamline data.

---

## 3.2 Working Principle

The generator simulates events as follows:

- **Beam Core Generation**:  
Particles forming the main Gaussian-shaped core of the beam are generated according to a 2D Gaussian distribution in the transverse (x, y) plane.

- **Halo Particle Injection**:  
Particles belonging to the beam halo are generated with a different radial distribution — typically broader and heavier-tailed — to mimic the halo effect caused by scattering or beam instabilities.

- **Noise Addition**:  
Random electronic or detector noise is added to simulate realistic imperfect readings.

- **Label Assignment**:  
Each generated point is labeled either as:
  - Core (Label 0), or
  - Halo (Label 1).

- **Parameter Variations**:  
The generator can dynamically adjust parameters such as:
  - Beam width (standard deviation of the Gaussian core)
  - Halo intensity (fraction of particles belonging to the halo)
  - Noise level (electronic noise simulation)

---

## 3.3 Mathematical Formulation

### 3.3.1 Core Particles

The (x, y) positions of core beam particles follow a normal distribution centered at the beam axis:

\[
x_{\mathrm{core}}, y_{\mathrm{core}} \sim \mathcal{N}(0, \sigma_{\mathrm{core}}^2)
\quad \text{(Eq. 9)}
\]

Where:
- \( \sigma_{\mathrm{core}} \) is the standard deviation of the beam core, controlling its width.

---

### 3.3.2 Halo Particles

Halo particles, formed by scattering, are modeled with a radial Cauchy distribution for distance and a uniform angular distribution:

\[
r_{\mathrm{halo}} \sim \text{Cauchy}(\gamma); \quad \theta \sim \text{Uniform}(0,2\pi)
\quad \text{(Eq. 10)}
\]

The x and y coordinates of halo particles are calculated from their radial distance and angle:

\[
x_{\mathrm{halo}} = r_{\mathrm{halo}}\cos(\theta), \quad y_{\mathrm{halo}} = r_{\mathrm{halo}}\sin(\theta)
\quad \text{(Eq. 11)}
\]

Where:
- \( \gamma \) controls the spread of the halo, determining how far particles scatter.
- \( \theta \) is the angle in the x-y plane, uniformly distributed to ensure circular symmetry.

---

### 3.3.3 Noise Addition

To account for detector imperfections, small Gaussian noise is added to each coordinate:

\[
x_{\mathrm{final}} = x + \epsilon_x, \quad y_{\mathrm{final}} = y + \epsilon_y
\quad \text{(Eq. 12)}
\]

Where:
- \( \epsilon_x, \epsilon_y \sim \mathcal{N}(0, \sigma_{\mathrm{noise}}^2) \) represent independent noise terms.
- \( \sigma_{\mathrm{noise}} \) is the standard deviation of the noise, reflecting detector resolution limits.

---

## 3.4 Sample Generated Data

| Event ID | S1 Time (ns) | S2 Time (ns) | DWC1 X (mm) | DWC1 Y (mm) | DWC2 X (mm) | DWC2 Y (mm) | DWC3 X (mm) | DWC3 Y (mm) | DWC4 X (mm) | DWC4 Y (mm) | Halo Q1 | Halo Q2 | Halo Q3 | Halo Q4 | Cherenkov 1 | Cherenkov 2 | CALO Energy (MeV) | Particle Type | True Energy (GeV) | Beam Stability | Halo Pattern |
|:--------:|:------------:|:------------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|:-------:|:-------:|:-------:|:-------:|:------------:|:------------:|:----------------:|:-------------:|:----------------:|:--------------:|:------------:|
| 1 | 0.003 | 0.029 | 2.1 | -1.8 | 2.5 | -1.5 | 2.6 | -1.7 | 2.7 | -1.6 | 120 | 90 | 110 | 85 | 1 | 0 | 5.1 | Electron | 5.0 | Stable | Symmetric |
| 2 | 0.002 | 0.030 | -0.5 | 1.3 | -0.7 | 1.8 | -0.8 | 1.7 | -0.9 | 1.6 | 45 | 120 | 70 | 140 | 0 | 1 | 2.8 | Pion | 8.0 | Unstable | Skewed |

(Only 2 sample rows shown; full dataset contains thousands of events.)

---

## 3.5 Adjustable Parameters

| Parameter | Description | Typical Values |
|:----------|:------------|:---------------|
| Core Standard Deviation \( \sigma_{\mathrm{core}} \) | Width of core beam spread | 1–3 mm |
| Halo Intensity | Fraction of halo particles | 5%–15% |
| Halo Spread \( \gamma \) | Broadness of halo radial distribution | 3–7 mm |
| Noise Standard Deviation \( \sigma_{\mathrm{noise}} \) | Detector/electronic noise magnitude | 0.1–0.5 mm |
| Number of Particles | Total number of events per sample | 5,000–50,000 particles |

---

# Model

---

## 4.1.1 Overview

Our model is designed to solve four distinct beamline diagnostic tasks using multi-task deep learning.  

The architecture is modular, multi-branch, and built for interpretability — supporting both scientific accuracy and physical explainability.

## 4.1.2 Why Neural Networks?

We chose a deep neural network as our primary model for several reasons:

Multimodal Inputs: The data includes diverse types — numerical (ToF, energy), categorical (Cherenkov hits), geometric (angular deviation), and spatial (halo quadrant patterns). Neural networks handle such mixed inputs effectively.

Hidden Nonlinearities: Beam-halo relationships are not trivially separable in linear space. NN layers help extract higher-level representations and correlations.

Multi-task Synergy: Tasks such as beam composition and energy regression are strongly interlinked. A shared backbone enhances learning for all tasks.

Scalability: Additional detectors or derived features (e.g., instability scoring) can be easily appended to the architecture.

---

## 4.2 Supported Tasks

| Task | Type | Output | Description |
|------|------|--------|-------------|
| **Task 1** | Classification (Multi-label) | Particle composition percentages | Predicts the proportion of electrons, pions, and protons in each beam spill using global features. |
| **Task 2** | Regression | Mean energy, position offset, and beam spread | Learns to predict physical beam properties from raw and derived detector inputs. |
| **Task 3** | Regression or Binary Classification | Beam stability index | Estimates time-resolved instability or classifies unstable vs. stable beams based on fluctuations. |
| **Task 4** | Classification | Halo shape class | Predicts beam halo pattern class (e.g., symmetric, skewed, multi-lobed). |

Each task uses dedicated output heads that are trained jointly via a shared feature backbone.

---

## 4.3 Input Features

Inputs are grouped by detector and role:

- **Halo Detector (HD):**  
  - 4 quadrant ADC values  
  - Asymmetry metrics  
  - Ratios (e.g., vertical/horizontal, diagonal symmetry)  
  - Attention applied on Halo inputs

- **ToF (S1–S2):**  
  - Delta time  
  - Derived beta, mass if available

- **DWC Tracking (1–4):**  
  - Angular deflection pre-/post-foil  
  - Beam offset  
  - Trajectory slope

- **Cherenkov 1 & 2:**  
  - Binary threshold flags  
  - Pressure-tuned PID signal

- **CALO (Lead Glass):**  
  - Energy deposition  
  - Used to distinguish EM vs. hadronic events

All inputs are normalized or standardized during preprocessing.

---

## 4.4 Model Architecture

### Modular Design

- **Input Branches** for each detector group  
  (e.g., `halo_branch`, `tracking_branch`, `pid_branch`, `calo_branch`)

- **Shared Dense Layers** after concatenation (hidden representation)

- **Four Output Heads**, each with its own activation and loss:
  - Softmax (composition)
  - Linear (regression)
  - Sigmoid (binary stability if needed)
  - Softmax (shape classification)

### Loss Function

A custom weighted multi-task loss:

\[
\mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{composition}} + \lambda_2 \mathcal{L}_{\text{energy}} + \lambda_3 \mathcal{L}_{\text{stability}} + \lambda_4 \mathcal{L}_{\text{halo-shape}}
\]

Weights \( \lambda_i \) are adjustable to prioritize specific tasks.

---

## 4.5 Attention on Halo Features

An attention mechanism is applied on the four halo quadrants and derived features.  
It learns which quadrant or asymmetry patterns contribute most to predicting beam composition or instability.

This improves the model’s ability to infer subtle differences in beam structure using coarse spatial data.

---

## 4.6 Output Structure

Each forward pass produces:

- A vector of composition probabilities  
- Regression values: [mean energy, x-offset, y-offset, beam spread X, beam spread Y]  
- Stability score or label  
- Halo shape class

These outputs can be analyzed individually or combined into a single dashboard/report.

---

## 4.7 Training and Evaluation

- **Optimizer:** Adam  
- **Losses:** Categorical CrossEntropy, MSE, Binary CrossEntropy  
- **Metrics:** Accuracy, MAE, R², Confusion Matrix, ROC AUC

Model training is logged and results are saved per task.  

---

## 4.8 Flexibility

This model is designed in a modular way and can be adapted to Add new tasks & functions(e.g., anomaly detection, beam spill classification)

---

# Simulation Results

Coming soon ...

---

# Contingency (Backup) Plan

---

## 6.1 Identified Risks and Mitigation Strategies

| Risk | Impact | Mitigation Strategy |
|:----|:------|:--------------------|
| **Tracking Detector Failure (DWC1–4)** | Loss of fine particle trajectory measurements | Focus on Time-of-Flight (ToF) and Cherenkov PID features. Use S1–S2 timing, halo asymmetries, and CALO signals to proceed with composition classification (Task 1) and halo-based tasks (Task 4). |
| **Halo Detector Malfunction (Partial or Complete)** | Loss of direct halo quadrant signals | Shift emphasis to DWC scattering angles and Cherenkov PID for classification. If needed, reframe Tasks 1 & 2 around core beam parameters only. |
| **S1 or S2 Scintillator Failure** | Loss of trigger and/or ToF measurement | Use only tracking and PID detectors for event reconstruction. Rely on Halo Quadrants + DWC angular spread for analysis. |
| **Cherenkov Detector Failure (C1 or C2)** | Incomplete PID (Particle ID) | Reinforce classification models using ToF and calorimeter features. Compensate particle separation loss through halo asymmetry patterns if statistically viable. |
| **Beam Composition Unexpected** | Different mixture than assumed (e.g., wrong proton/electron ratios) | Retrain machine learning models on early collected data. Adjust experimental goals dynamically to focus on available particle types. |
| **Beam Energy Instability or Low Statistics** | Reduced event quality and statistical power | Prioritize data collection for baseline runs (no foil, standard beam) and carbon foil runs first. Limit variations to maximize usable datasets for Tasks 1 and 2. |
| **DAQ Issues or Synchronization Problems** | Incomplete datasets or timing errors | Use redundant signals (S1 and Halo hits) to validate events. Perform offline corrections if timing skew is systematic. |
| **Beam Alignment Offsets** | Beam shifted from center, affecting halo symmetry | Correct for beam offsets offline during data reconstruction. If impossible, use relative quadrant differences instead of absolute patterns. |
| **Magnet (if requested) Not Available** | No direct momentum measurement | Focus on relative energy spread via halo width and calorimeter response instead of exact momentum estimation. Omit magnet-based tasks if necessary. |
| **Limited Beamtime or Interruptions** | Fewer datasets than planned | Strict run prioritization:  
  1. Baseline runs (no foil)  
  2. Carbon foil runs  
  3. Aluminum foil runs  
  4. Copper foil or collimator runs  
Only attempt different energies if main conditions are satisfied. |

---

## 6.2 Strategy

- **Prioritize critical physics goals** first (Tasks 1 and 4: composition classification and halo pattern recognition).
- **Remain flexible**: focus dynamically on the most robust detectors available at any time.
- **Minimize dependence** on heavily correlated multi-detector events if failures occur.

---

# Final Note

---

It may be noted that this project has been formally submitted to the Beamline for Schools (BL4S) 2025 competition.  
In the event of selection as one of the top teams and gaining the opportunity to carry out the experiment at CERN,  
all details of this experience, as well as the results, will be documented and published in this same repository.



