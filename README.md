# Fraud-Detection-Transactions

This project focuses on detecting fraudulent transactions using machine learning techniques. It includes exploratory data analysis, feature engineering, model training with XGBoost, hyperparameter tuning, and evaluation using various metrics (e.g., precision, recall, ROC-AUC, and confusion matrix).

The goal of this project is to build a predictive model that can accurately identify fraudulent transactions. We use a combination of data preprocessing, exploratory data analysis (EDA), and machine learning—primarily leveraging the XGBoost algorithm—to address the problem.

# Dataset

Source: The dataset used in this project is assumed to contain transaction data with features that include both numerical and categorical variables.
Target Variable: The Fraud_Label column represents whether a transaction is fraudulent (1) or not (0).
Features: The dataset contains a mix of numeric and string features, which are analyzed and engineered to improve model performance.

https://www.kaggle.com/datasets/samayashar/fraud-detection-transactions-dataset

# Data Exploration and Preprocessing:
Load and inspect the dataset.
Filter and separate numeric features using Pandas’ select_dtypes.
Handle categorical data by mapping string values to numeric codes.
Explore feature distributions (e.g., quantile analysis for continuous variables).
Exploratory Data Analysis (EDA):
Generate plots to visualize the predictive power of features.
Compare feature distributions between fraudulent and non-fraudulent transactions.
Use visualization tools such as Matplotlib and Seaborn to analyze trends.

# Feature Engineering:
Create quantiles for continuous variables to analyze predictive power.
Map categorical features to numeric values where necessary.
Investigate correlations and relationships between features.
Modeling and Evaluation

# Modeling:
An XGBoost classifier is used to model the problem. The model is trained on the dataset after splitting it into training and test sets.
Cross Validation:
Cross validation (using KFold or StratifiedKFold) is employed on the training set to obtain reliable performance estimates.
Evaluation Metrics:
Model performance is evaluated using metrics such as:
Accuracy
ROC-AUC
Precision and Recall
Confusion Matrix
Hyperparameter Tuning

Hyperparameter tuning is performed using scikit-learn's RandomizedSearchCV to optimize the XGBoost model. Key parameters tuned include:

learning_rate
max_depth
min_child_weight
subsample
colsample_bytree
gamma
reg_alpha
reg_lambda

A randomized search approach is used to reduce computational time by testing a subset of all possible parameter combinations.
