# Fraud Detection using Machine Learning

Problem Statement
A financial company wants to proactively detect fraudulent transactions to minimize financial loss and protect customers. The task is to develop a machine learning model that can accurately identify whether a transaction is fraudulent or not.

# This challenge involves:

Handling highly imbalanced data

Performing statistical analysis

Applying feature engineering

Using ML models to generate insights and actionable intelligence

# Dataset Overview
Source: Provided in CSV format

Size: 6,362,620 rows × 10 columns

Target Variable: isFraud (0 = Legitimate, 1 = Fraudulent)

Key Features:

step: Time step of the transaction

type: Type of transaction (e.g., CASH_OUT, PAYMENT)

amount: Transaction amount

oldbalanceOrg, newbalanceOrig: Originator account balances before and after the transaction

oldbalanceDest, newbalanceDest: Destination account balances before and after

isFraud: Indicator of a fraudulent transaction

isFlaggedFraud: Indicator of system flagging (not reliable)

# Exploratory Data Analysis (EDA)
The dataset is heavily imbalanced:

Legitimate: 6,354,407

Fraudulent: 8,213

Certain transaction types (e.g., TRANSFER, CASH_OUT) show disproportionately high fraud.

Fraudulent transactions often have sudden changes in balances.

 Feature Engineering
A key feature was engineered:

balance_delta_orig = oldbalanceOrg - newbalanceOrig
This feature captured sudden drops in the originator’s balance, a strong indicator of potential fraud.

Feature importance score: 0.5 (highest among all features)
This highlights that domain-specific feature engineering significantly improved model performance.

⚙️ Preprocessing
Label Encoding was applied to the type column (sufficient for tree-based models).

No scaling was applied since models like Random Forest and XGBoost are scale-invariant.

SMOTE (Synthetic Minority Over-sampling Technique) was used to balance the dataset and address class imbalance.

# Model Building
Three models were trained and compared:

Logistic Regression

Random Forest Classifier

XGBoost Classifier

Evaluation metrics:

Precision

Recall

F1-Score

ROC-AUC Score

Confusion Matrix

# Model Evaluation Results
📉 Logistic Regression (Baseline)
Poor performance
Couldn't handle non-linearities or class imbalance effectively

Very low precision and recall for fraud class

🌲 Random Forest
Recall (fraud): 93.8%

Precision (fraud): 2%

ROC-AUC: 0.985

High number of false positives

 # 🚀 XGBoost (Best Performing)
Recall (fraud): 99.38%

Precision (fraud): 14.37%

F1-Score (fraud): 0.25

ROC-AUC: 0.9991

Significantly fewer false positives and best balance between precision and recall

# 🧠 Key Insights & Achievements
Custom feature balance_delta_orig had the highest importance (0.5), showing deep understanding of the transaction logic and its role in fraud.

SMOTE oversampling effectively addressed class imbalance, enabling better learning on the minority class.

XGBoost outperformed other models by a wide margin, achieving near-perfect AUC and high recall.

Threshold optimization was done to enhance performance based on business need (recall > precision).

# 🏁 Conclusion
This project demonstrates how combining:
Strong EDA
Thoughtful feature engineering
Class balancing
Model tuning
...can lead to powerful predictive models for fraud detection.

The deployed XGBoost model not only identifies nearly all fraudulent transactions but does so with a reasonable false positive rate — making it a practical choice for real-world deployment.

# 🔄 Future Work
Cost-sensitive learning to further reduce false positives

Real-time fraud detection pipeline using model

Ensemble or stacking models to push boundaries further

Add explainability tools like SHAP for feature impact visualization
