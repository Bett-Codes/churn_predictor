# 📉 Customer Churn Prediction — End-to-End ML Project
A full end-to-end machine learning project that predicts whether a telecom customer will churn, complete with an interactive Streamlit web app and SHAP-powered explainability.

## Project Overview

Customer churn is one of the most critical business problems in the telecom industry. This project builds a complete ML pipeline — from raw data to a deployed interactive app — that helps business teams identify at-risk customers and take proactive retention actions.

## Project Structure

```
churn-predictor/
│
├── app.py                                    # Streamlit web application
├── churn_prediction.py                       # Full ML pipeline (Jupyter-ready)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv     # Dataset (Telco Customer Churn)
└── README.md                                 # Project documentation
```

##  ML Pipeline

### 1. Data Cleaning
- Fixed `TotalCharges` dtype (spaces stored as nulls)
- Dropped 11 rows with missing values
- Removed non-feature column `customerID`
- Encoded target `Churn` → binary (Yes=1, No=0)

### 2. Exploratory Data Analysis
- Churn distribution → ~26% churn rate (imbalanced dataset)
- KDE plots for numeric features by churn status
- Churn rate bar charts across contract type, payment method, internet service
- Correlation heatmap

### 3. Feature Engineering
| Feature | Description |
|---|---|
| `AvgMonthlySpend` | TotalCharges / (tenure + 1) |
| `TenureBucket` | Binned tenure: 0-1yr, 1-2yr, 2-4yr, 4+yr |

- Label encoding for binary columns
- One-hot encoding for multi-class categoricals
- StandardScaler for numeric features
- SMOTE to handle class imbalance

### 4. Models Trained
| Model | Notes |
|---|---|
| Logistic Regression | Baseline |
| Random Forest | 200 estimators |
| Gradient Boosting | 200 estimators |
| XGBoost | Best performer |

- Evaluated using ROC-AUC, confusion matrix, precision/recall/F1
- Best model selected automatically by AUC score

### 5. Explainability
- SHAP TreeExplainer for global + local feature importance
- Beeswarm plot, bar chart, and per-customer force plots

## Streamlit App Features

| Tab | Contents |
|---|---|
| Predict Churn | Customer input form → churn probability gauge → local SHAP explanation |
| Model Performance | AUC score cards, ROC curves, confusion matrix, classification report |
| SHAP Explainability | Global beeswarm plot, feature importance bar chart, key insights |

## Key Insights

- Tenure is the strongest churn predictor — newer customers churn far more
- Month-to-month contracts significantly increase churn probability
- High monthly charges without added-value services drive customers away
- Customers without Online Security or Tech Support are at higher risk

## Tech Stack

- Data: Pandas, NumPy
- Visualization: Matplotlib, Seaborn
- ML: Scikit-learn, XGBoost, imbalanced-learn
- Explainability: SHAP
- App: Streamlit
- Deployment: Streamlit Cloud
 
## Dataset

Telco Customer Churn — IBM Sample Dataset  
Source: [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- 7,043 customer records
- 21 features covering demographics, services, billing & contract info
