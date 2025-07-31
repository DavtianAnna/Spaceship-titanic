# 🛸 Spaceship Titanic - Kaggle Classification Challenge

This repository contains my solution for the [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic) Kaggle competition.

The main goal of this challenge is to predict whether a passenger was **transported to another dimension** (`Transported`) during a spaceship malfunction.  
This is a **binary classification** task where the target column is `Transported` (boolean: `True` or `False`).

---

## 🗂️ Project Structure

```
project-root/
├── 📄 titanic_task.ipynb       # Jupyter notebook containing the complete ML workflow: data preprocessing, model training, tuning, and evaluation .
├── 📊 train.csv                # Training dataset with features and target ('Transported').
├── 🧪 test.csv                 # Test dataset: same features, no target.
├── 📝 sample_submission.csv    # Template for Kaggle submission.
├── 🚀 submission1.csv          # Submission file from Logistic Regression model.
├── 🚀 submission2.csv          # Submission file from XGBoost model.
└── 📜 README.md                # Project documentation.

```

---

## 💻 Technologies Used

- **Language**: Python 3.x  
- **Environment**: Jupyter Notebook  
- **Libraries**:
  - Data Handling: `pandas`, `numpy`
  - Visualization: `seaborn`, `matplotlib`
  - Preprocessing: `LabelEncoder`, `StandardScaler`
  - ML Algorithms: `LogisticRegression`, `XGBClassifier`
  - Model Tuning: `Pipeline`, `GridSearchCV`
  - Evaluation: `classification_report`, `confusion_matrix`, `ConfusionMatrixDisplay`


---


## 📊 Dataset Description

The dataset is provided by Kaggle under the competition: [Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic).

- `train.csv`: Historical training data with the binary target `Transported`.
- `test.csv`: Test set without labels.
- `sample_submission.csv`: Format for predictions to submit to Kaggle.

---

## 🔁 Workflow Summary

The `titanic_task.ipynb` notebook includes the following steps:

### 1. Data Cleaning & Imputation
- Dropped irrelevant columns: `Name`, `Cabin`, `PassengerId`
- Filled missing values using random imputation based on the column’s value distribution.

### 2. Feature Encoding
- Applied `LabelEncoder` on all categorical and boolean features.

### 3. Model 1: Logistic Regression
- Used a pipeline with `StandardScaler` and `LogisticRegression`.
- Tuned using `GridSearchCV`:
  - Penalties: `l1`, `l2`, `elasticnet`, `none`
  - Solvers: `lbfgs`, `saga`, `liblinear`, etc.
- Evaluated with `classification_report` and confusion matrix.

### 4. Model 2: XGBoost Classifier
- Used `XGBClassifier` with hyperparameter tuning:
  - `n_estimators`, `max_depth`, `learning_rate`, `subsample`
- Selected as final model based on validation accuracy.

### 5. Prediction & Submission
    - Both models predict class probabilities on test data.
    - Outputs saved as:
        - `submission1.csv` → Logistic Regression
        - `submission2.csv` → XGBoost


---

## 📈 Results Summary

| Model               | Tuning Method            | Evaluation Metric        | Output File         |
|--------------------|--------------------------|---------------------------|----------------------|
| Logistic Regression | Pipeline + GridSearchCV  | Accuracy, F1-score        | `submission1.csv`    |
| XGBoost Classifier  | GridSearchCV             | Accuracy, F1-score        | `submission2.csv`    |



## ⚙️ Installation

To install all required dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost


