# 📘 Loan Recovery System

## Overview

The **Loan Recovery System** is a modular, machine-learning-enhanced framework designed to optimize loan collection strategies. It provides a Streamlit interface for users to upload data, perform analysis, and visualize actionable insights across borrower risk segmentation, pricing optimization, anomaly detection, and more.

## 📂 Project Structure

```
.
├── loanRecoverySystemApp.py            # Streamlit interface for UI and analysis modules
├── advancedstrategies.py              # Advanced modeling: survival analysis, neural nets, dynamic pricing
├── optimisingLoanRecovery.py          # Base strategy model for risk segmentation and recovery optimization
```

## 🔧 Components & Features

### 1. `LoanRecoveryStrategyModel` (Base Class)

Located in: `optimisingLoanRecovery.py`

#### Key Functions

- **`load_and_prepare_data(df)`**: Cleans, encodes, and scales input data.
- **`segment_borrowers(n_clusters=4)`**: Uses KMeans to segment borrowers into risk groups.
- **`train_recovery_optimization_model()`**: Trains a Random Forest classifier to predict optimal collection methods.
- **`develop_early_warning_system()`**: Uses Gradient Boosting to predict default risk.
- **`calculate_roi_optimization()`**: Calculates ROI for various collection strategies.
- **`predict_optimal_strategy(borrower_data)`**: Suggests the best method for a given borrower.

### 2. `AdvancedLoanRecoverySystem` (Extended Class)

Located in: `advancedstrategies.py`  
Inherits from `LoanRecoveryStrategyModel`

#### Advanced Capabilities

- **`assign_risk_segments()`**: Assigns risk tiers based on payment behavior.
- **`implement_survival_analysis()`**: Predicts time-to-recovery using Cox Proportional Hazards modeling.
- **`detect_anomalous_cases()`**: Uses Isolation Forest to flag unusual borrower patterns.
- **`implement_neural_predictor()`**: Neural network for classifying borrower risk.
- **`dynamic_pricing_optimization()`**: Determines optimal settlement amounts using profit-based optimization.
- **`analyze_seasonality_patterns()`**: Simulates and visualizes seasonal recovery trends.
- **`implement_network_analysis()`**: Analyzes borrower interconnections and how network position affects risk.
- **`create_recovery_dashboard()`**: Generates a multi-panel dashboard visualizing KPIs.
- **`generate_action_recommendations()`**: Provides strategy recommendations based on analysis.

### 3. `StreamlitLoanRecoverySystem` (Streamlit Integration)

Located in: `loanRecoverySystemApp.py`

#### Streamlit Pages

- **Dashboard**: Visual KPIs like total accounts, avg loan, high-risk %, etc.
- **Risk Analysis**: View risk segment distributions and correlations.
- **Anomaly Detection**: Interactive detection and visualization of anomalous loans.
- **Neural Network**: Train and display performance of neural classifier.
- **Pricing Optimization**: Visual and tabular outputs of dynamic settlement suggestions.
- **Seasonality**: Explore seasonal patterns in recovery rates.
- **Network Analysis**: Visualize and analyze borrower relationship graphs.
- **Recommendations**: Actionable insights based on analysis results.

## 📊 Data Requirements

Minimum recommended columns:

- `Loan_Amount`
- `Num_Missed_Payments`
- `Days_Past_Due`
- `Contact_Success_Rate`
- `Previous_Recovery_Rate`
- `Credit_Score`
- `Employment_Length`
- `Income`
- `Borrower_Age`

> Missing features are synthetically generated for demo purposes if not present.

## 🧠 Machine Learning Models

| Feature                      | Algorithm Used           | Purpose                              |
| ---------------------------- | ------------------------ | ------------------------------------ |
| Borrower Segmentation        | KMeans Clustering        | Identify risk clusters               |
| Recovery Strategy Prediction | Random Forest            | Suggest optimal recovery method      |
| Default Risk Prediction      | Gradient Boosting        | Early warning for likely defaults    |
| Survival Analysis            | Cox Proportional Hazards | Estimate time to recovery            |
| Anomaly Detection            | Isolation Forest         | Identify irregular borrower patterns |
| Risk Classification          | MLP Neural Network       | Predict borrower risk tier           |

## 📈 Visual Outputs

- PCA Clustering plots
- Survival curves by risk
- Heatmaps for ROI and pricing
- Anomaly score distributions
- Recovery trends (monthly/weekly/quarterly)
- Network centrality and clustering plots
- Multi-panel dashboard with ROI, anomalies, cost-benefit etc.

## ⚙️ How to Use

### Run Locally

```bash
streamlit run loanRecoverySystemApp.py
```

### Upload Your Data

- Supported formats: `.csv`, `.xlsx`, `.xls`
- Or generate synthetic demo data in-app

## ✅ Recommendations Engine

Based on all model outputs, the system generates:

- Risk-adjusted strategies per borrower
- Optimal recovery methods
- Expected ROI and time-to-resolution
- Anomaly alerts
- Seasonal timing suggestions

## 📌 Notes

- All models use synthetic enhancements if necessary (e.g., missing target columns)
- Modular architecture allows easy integration of new strategies or data
- Visualization is optimized for interpretability and executive decision-making
