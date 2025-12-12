# ---------------------------------------------
# FRAUD DETECTION MODEL - UPGRADED VERSION
# Author: Kishore K
# ---------------------------------------------

# 1. IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import seaborn as sns
import joblib

# 2. LOAD DATA
# Replace with your dataset path
df = pd.read_csv("synthetic_fraud_dataset.csv")

print("Dataset shape:", df.shape)
print(df.head())

# 3. CHECK MISSING VALUES
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

# 4. SEPARATE FEATURES & TARGET
X = df.drop("Class", axis=1)    # Change 'Class' if your target column name differs
y = df["Class"]

# 5. FEATURE SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42, stratify=y
)

# 7. HANDLE IMBALANCED DATA USING SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts())
print("After SMOTE:", y_res.value_counts())

# 8. RANDOM FOREST MODEL
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_res, y_res)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# 9. XGBOOST MODEL
xgb_model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)

xgb_model.fit(X_res, y_res)

xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

# 10. EVALUATION FUNCTION
def evaluate_model(name, y_true, y_pred, y_prob):
    print(f"\n----- {name} MODEL -----")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    print("ROC-AUC:", roc_auc_score(y_true, y_prob))

# 11. SHOW SCORES
evaluate_model("Random Forest", y_test, rf_pred, rf_prob)
evaluate_model("XGBoost", y_test, xgb_pred, xgb_prob)

# 12. CONFUSION MATRIX
cm = confusion_matrix(y_test, xgb_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 13. ROC CURVE
RocCurveDisplay.from_predictions(y_test, xgb_prob)
plt.title("ROC Curve - XGBoost")
plt.show()

# 14. SAVE THE BEST MODEL
joblib.dump(xgb_model, "fraud_detection_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved as fraud_detection_model.pkl")
print("Scaler saved as scaler.pkl")
