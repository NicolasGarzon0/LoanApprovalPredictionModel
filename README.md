# 🏦 Loan Approval Prediction Model — Machine Learning Pipeline

A complete machine learning pipeline to predict whether a loan application should be **approved**.

This project walks through **EDA**, **preprocessing**, **model training**, **hyperparameter tuning**, **explainability**, and **evaluation**, with a focus on maximizing performance and interpretability using models like **LightGBM** or **CatBoost**.

---

## 🔍 Objective

Build a supervised learning model to predict loan approval outcomes using structured financial data. 

---

## 📊 Dataset

The project uses `train.csv` from the [Kaggle Playground Series - S4E10](https://www.kaggle.com/competitions/playground-series-s4e10/data), a synthetic dataset modeled on real loan approval data. 

---

## 🧪 Steps Overview

### 1. **EDA & Data Understanding**
- Explored applicant demographics, income, credit history, loan intent, among other features of the dataset.
- Identified correlations between features and loan approval status.

### 2. **Preprocessing**
- Cleaned and standardized categorical variables.
- Performed binary and one-hot encoding.
- Prepared a minimally processed dataset that yielded the best model performance.

### 3. **Baseline Modeling**
- Trained a Logistic Regression model as the baseline.
- Established baseline performance ROC AUC.

### 4. **Advanced Modeling**
- Trained and evaluated: Random Forest, Gradient Boosting, CatBoost, and LightGBM models.
- Compared models based on **ROC AUC** scores to select the best-performing model for further enhancement.

### 5. **Hyperparameter Tuning**
- Tuned LightGBM using Optuna and also explored SHAP-based feature selection.
- Selected the Optuna-optimized model, as SHAP-based selection did not significantly improve performance.
- Achieved best AUC of **0.9603**.

### 6. **Model Evaluation & Explainability**
- Final evaluation of the model on the test set using:
  - ROC AUC, Accuracy, Precision, Recall, F1 Score
  - Confusion matrix heatmap
  - ROC Curve visualization

---

## 🎯 Model Performance (LightGBM)

| Metric     | Score     |
|------------|-----------|
| ROC AUC    | 0.9603    |
| Accuracy   | 95.23%    |
| Precision  | 92.17%    |
| Recall     | 72.69%    |
| F1 Score   | 81.29%    |

---

## 🛠️ Tech Stack

- `Python`
- `Pandas`, `NumPy` — data manipulation, cleaning, and numerical computations
- `Scikit-learn` — preprocessing, model training, evaluation
- `LightGBM`, `CatBoost`, `Gradient Boosting`, `Random Forest` — ML Algorithms
- `Optuna` — hyperparameter tuning
- `Matplotlib`, `Seaborn` — data visualization
- `Joblib` — saving and loading trained models

---

## 📦 Setup

1. **Clone the repository**
```bash
git clone https://github.com/NicolasGarzon0/LoanApprovalPredictionModel.git
cd LoanApprovalPredictionModel
```

2. **Install requirements**
```bash
pip install -r requirements.txt
```

3. **Run notebooks step-by-step**

Each notebook builds upon the last. Start from `01_eda.ipynb`.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).

