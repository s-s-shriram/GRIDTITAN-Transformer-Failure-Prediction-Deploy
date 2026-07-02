⚡ GRIDTITAN

AI-Powered Transformer Risk Intelligence System

GRIDTITAN is a Data-Centric AI-powered Transformer Risk Intelligence System designed to predict transformer failures, identify high-risk assets, and support proactive maintenance planning in power distribution networks.

The project focuses on improving data quality through KNN Imputation, Feature Engineering, and SMOTE balancing before model training, enabling reliable transformer failure prediction and risk intelligence generation.

---

🚀 Project Overview

Transformer failures are one of the major causes of power outages, equipment damage, and maintenance challenges in electrical distribution systems. Traditional maintenance approaches are often reactive, resulting in delayed fault detection and increased operational costs.

GRIDTITAN addresses this challenge by leveraging Data-Centric AI techniques to improve data quality and generate accurate transformer failure predictions. The system identifies high-risk transformers, provides maintenance recommendations, and supports proactive maintenance planning.

---

🎯 Objectives

- Predict transformer failure risk using AI.
- Improve failure detection capability.
- Identify high-risk transformers before breakdown.
- Support maintenance prioritization and planning.
- Reduce outage risk through proactive decision-making.
- Demonstrate the impact of Data-Centric AI in industrial applications.

---

🏗 System Architecture

Transformer Operational Data

↓
KNN Imputation

↓
Feature Engineering

↓
SMOTE Balancing

↓
XGBoost Prediction

↓
Risk Intelligence Engine

↓
Maintenance Recommendations

---

📊 Dataset

A realistic transformer operational dataset containing approximately 2000 records was developed to simulate transformer behavior.

Features

Feature| Description
transformer_id| Unique Transformer Identifier
load| Transformer Load
temperature| Operating Temperature
voltage| Voltage Reading
current| Current Reading
power| Power Consumption
failure| Failure Status (0/1)

---

🔍 Data Quality Challenges

The dataset contains several real-world challenges:

- Missing values
- Class imbalance
- Sensor noise
- Rare failure events
- Limited failure representation

These challenges significantly affect predictive model performance.

---

🧠 Data-Centric AI Pipeline

Instead of focusing only on model selection, GRIDTITAN improves data quality before model training.

1. KNN Imputation

Missing values are handled using K-Nearest Neighbors (KNN) Imputation.

Benefits:

- Preserves operational patterns
- Improves data completeness
- Reduces information loss

---

2. Feature Engineering

Engineered features were created to capture hidden transformer stress conditions.

Features Created

- Thermal Stress
- Overload Indicator
- Load Ratio
- Voltage Deviation
- Current Stress
- Load Fluctuation

These features improve the model's ability to identify failure patterns.

---

3. SMOTE

Synthetic Minority Oversampling Technique (SMOTE) was applied to address class imbalance.

Benefits:

- Balances failure and non-failure classes
- Improves learning of rare failure events
- Enhances recall performance

---

4. XGBoost

XGBoost was selected as the final prediction model because it:

- Handles structured tabular data effectively
- Captures nonlinear relationships
- Performs well on noisy industrial datasets
- Provides strong failure detection capability

---

🤖 Machine Learning Models

Baseline Model

Logistic Regression

Results:

- Accuracy: 94%
- Recall: 0%

Although accuracy was high, the model failed to identify transformer failures.

---

Improved Model

XGBoost Classifier

Results:

- Accuracy: 89%
- Recall: 91%

The improved Data-Centric AI pipeline significantly enhanced failure detection performance.

---

📈 Performance Comparison

Metric| Baseline| Improved
Model| Logistic Regression| XGBoost
Accuracy| 94%| 89%
Recall| 0%| 91%

Key Insight

Improving data quality had a greater impact on failure detection than simply changing the machine learning model.

---

🖥 Key Features

Transformer Risk Prediction

Predicts failure probability using transformer operational parameters.

Risk Intelligence Dashboard

Provides visual risk assessment and monitoring.

Risk Classification

Classifies transformers into:

- Low Risk
- Medium Risk
- High Risk

Maintenance Recommendations

Generates actionable maintenance suggestions.

Maintenance Priority Ranking

Identifies high-risk transformers requiring immediate attention.

Downloadable Reports

Supports maintenance planning through downloadable risk reports.

---

🌍 Real-World Applications

- Power Distribution Utilities
- Smart Grid Systems
- Transformer Asset Management
- Predictive Maintenance Programs
- Industrial Power Monitoring

---

🔮 Future Scope

Future enhancements include:

- SCADA Integration
- Real-Time Transformer Monitoring
- Live Risk Heatmaps
- Failure Probability Forecasting
- Explainable AI Dashboard
- Smart Alert Systems
- Smart Grid Deployment

---

🛠 Technology Stack

Frontend

- Streamlit

Backend

- Python

Machine Learning

- Scikit-Learn
- XGBoost
- Imbalanced-Learn

Data Processing

- Pandas
- NumPy

Visualization

- Matplotlib
- PyDeck

---

📂 Project Structure

GRIDTITAN/
│
├── app.py
├── requirements.txt
├── transformer_dataset.csv
├── model.pkl
├── README.md
│
├── images/
│
├── reports/
│
└── outputs/

▶ Installation

Clone Repository

git clone [https://github.com/s-s-shriram/GRIDTITAN-Transformer-Failure-Prediction.git]

cd GRIDTITAN

Install Dependencies

pip install -r requirements.txt

Run Application

streamlit run app.py

---

🏆 Achievement

Top 10 Finalist

TNSDC Naan Mudhalvan Advanced AI/ML Hackathon 2026

GRIDTITAN was selected as a Top 10 Finalist among competing teams for demonstrating a strong Data-Centric AI approach and practical transformer failure prediction solution.

---

👥 Team GRIDTITAN

SHRIRAM S S

Project Lead & AI/ML Architect

Led project planning, AI pipeline design, machine learning development, system architecture, and final project integration.

HARIHARA SUBRAMANIAN K

Data Engineering & Feature Engineering

Handled data preprocessing, KNN Imputation, feature engineering, and dataset preparation.

SHESHANTH R

Machine Learning Modeling & Optimization

Developed and optimized predictive models, evaluated performance metrics, and supported risk scoring.

SANJAY S

Dashboard Development & Technical Documentation

Supported dashboard development, project documentation, testing, and presentation preparation.

---

🎯 Team Vision

Team GRIDTITAN is committed to leveraging Data-Centric AI, predictive analytics, and intelligent decision-support systems to improve power distribution reliability and enable proactive transformer maintenance.

---

📌 Conclusion

GRIDTITAN demonstrates how Data-Centric AI can significantly improve transformer failure prediction by addressing data quality challenges before model training.

By combining KNN Imputation, Feature Engineering, SMOTE balancing, and XGBoost prediction, the system improved failure detection recall from 0% to 91%, enabling proactive maintenance intelligence and reducing transformer failure risk.

⚡ GRIDTITAN – Predicting Failures Before They Happen
