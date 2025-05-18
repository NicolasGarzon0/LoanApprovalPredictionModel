import lightgbm as lgb
import seaborn as sns
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from catboost import CatBoostClassifier
from lightgbm import early_stopping, log_evaluation
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    roc_curve
)

def evaluate_model_on_dataset(dataset, model_obj):
    X = dataset.drop(columns=["loan_status"])
    y = dataset["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = clone(model_obj)
    model.fit(X_train, y_train)

    y_scores = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_scores)

    return auc

def evaluate_catboost_on_dataset(df, target_col, categorical_cols=None):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    cat_features = [col for col in (categorical_cols or []) if col in X.columns]

    model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.05,
        depth=6,
        eval_metric='AUC',
        verbose=0,
        random_state=42
    )

    model.fit(X_train, y_train, cat_features=cat_features)

    y_scores = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_scores)
    
    return auc

def evaluate_lightgbm_on_dataset(df, target_col, params=None):
    # Hardcoded config
    test_size = 0.2
    random_state = 42
    drop_cols = ["id"]

    # Prepare features and labels
    X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
    y = df[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Default LightGBM parameters if not provided
    if params is None:
        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "n_estimators": 200,
            "max_depth": 6,
            "random_state": random_state,
            "verbose": -1
        }

    # Train model
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[
            early_stopping(20, verbose=False),
            log_evaluation(0)
        ]
    )

    # Evaluate
    y_probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_probs)

    return auc

def evaluate_model(model, X_test, y_test, model_name):
    input_cols = joblib.load("models/input_columns.pkl")
    X_test = X_test[input_cols]

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    print(f"=== {model_name} Evaluation ===")
    print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc_score(y_test, y_pred_proba):.4f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve")
    plt.legend()
    plt.show()

    
def save_evaluation_summary(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    summary = {
        "Model": "LightGBM",
        "ROC AUC": roc_auc_score(y_test, y_pred_proba),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("reports/lightgbm_evaluation_summary.csv", index=False)
    return summary_df