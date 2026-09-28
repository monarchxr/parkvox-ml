"""
Voice-based Parkinson's Detection — Model Training & Ensemble
Nishka's part — PSIT Kanpur, Project 27_CS_DS_4C_10
"""

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ---- 1. Load data ----
df = pd.read_csv("parkinsons_cleaned.csv")

# Drop non-feature columns. 'name' and 'subject_id' are identifiers, not predictive features.
drop_cols = [c for c in ["name", "subject_id"] if c in df.columns]
df = df.drop(columns=drop_cols)

# The 22 features from Step 0 (exact order matters — this is the team contract)
FEATURE_NAMES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
    "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
    "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA",
    "spread1", "spread2", "D2", "PPE"
]

X = df[FEATURE_NAMES]
y = df["status"]

print(f"Dataset shape: {X.shape}, Class balance: {y.value_counts().to_dict()}")

# ---- 2. Train/test split + scaling ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- 3. Base models ----
base_models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42),
}

print("\n--- Individual base model performance ---")
individual_results = {}
for name, model in base_models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    individual_results[name] = {"accuracy": acc, "f1": f1}
    print(f"{name:22s} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

# ---- 4. Stacking ensemble ----
estimators = [
    ("lr", LogisticRegression(max_iter=2000, random_state=42)),
    ("svm", SVC(probability=True, random_state=42)),
    ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ("xgb", XGBClassifier(eval_metric="logloss", random_state=42)),
]

stacked_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=2000, random_state=42),
    cv=5,
)
stacked_model.fit(X_train_scaled, y_train)

# ---- 5. Evaluate stacked model ----
y_pred = stacked_model.predict(X_test_scaled)
y_proba = stacked_model.predict_proba(X_test_scaled)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

print("\n--- Stacked Ensemble Performance ---")
print(f"Accuracy    : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision   : {precision_score(y_test, y_pred):.4f}")
print(f"Recall (Sn) : {recall_score(y_test, y_pred):.4f}")
print(f"Specificity : {specificity:.4f}")
print(f"F1 Score    : {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC     : {roc_auc_score(y_test, y_proba):.4f}")
print(f"Confusion Matrix:\n[[TN={tn} FP={fp}]\n [FN={fn} TP={tp}]]")

# ---- 6. Save artifacts ----
joblib.dump(stacked_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(FEATURE_NAMES, "feature_names.pkl")

print("\nSaved: model.pkl, scaler.pkl, feature_names.pkl")
print("Ready to push to shared repo/Drive folder.")