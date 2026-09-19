"""
train_model.py
Trains the Bank Customer Churn XGBoost model and saves artifacts to /models.
Run this once locally, and it also runs automatically on Render at deploy time
(see the start command in render.yaml / Procfile).
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Churn_Modelling.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)


def train():
    print('Loading data...')
    df = pd.read_csv(DATA_PATH)

    # --- Preprocessing (mirrors the notebook) ---
    df_clean = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'])

    le_gender = LabelEncoder()
    df_clean['Gender'] = le_gender.fit_transform(df_clean['Gender'])  # Male=1, Female=0

    df_clean = pd.get_dummies(df_clean, columns=['Geography'], drop_first=True)
    bool_cols = df_clean.select_dtypes(include='bool').columns
    df_clean[bool_cols] = df_clean[bool_cols].astype(int)

    X = df_clean.drop(columns=['Exited'])
    y = df_clean['Exited']
    feature_names = X.columns.tolist()

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- SMOTE ---
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # --- Scale ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)
    X_test_scaled = scaler.transform(X_test)

    # --- Train XGBoost ---
    print('Training XGBoost model...')
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train_bal)

    # --- Quick sanity check on test set ---
    from sklearn.metrics import accuracy_score, roc_auc_score
    y_pred = xgb_model.predict(X_test_scaled)
    y_prob = xgb_model.predict_proba(X_test_scaled)[:, 1]
    print(f'Test Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(f'Test ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}')

    # --- Save artifacts ---
    joblib.dump(xgb_model, os.path.join(MODELS_DIR, 'xgb_model.pkl'))
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.pkl'))
    joblib.dump(le_gender, os.path.join(MODELS_DIR, 'gender_encoder.pkl'))

    print(f'Artifacts saved to {MODELS_DIR}:')
    print(os.listdir(MODELS_DIR))


if __name__ == '__main__':
    train()
