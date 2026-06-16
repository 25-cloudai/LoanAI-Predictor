import pandas as pd

# Load Dataset
df = pd.read_csv("data/loan.csv")

# Separate Features and Target
X = df.drop("loan_paid_back", axis=1)
y = df["loan_paid_back"]

# Encode Categorical Columns
from sklearn.preprocessing import LabelEncoder
import joblib

encoders = {}

categorical_columns = [
    'gender',
    'marital_status',
    'education_level',
    'employment_status',
    'loan_purpose'
]

for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

print(X.head())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

from imblearn.over_sampling import SMOTE

print("Before SMOTE:")
print(y_train.value_counts())

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(
    X_train,
    y_train
)

print("\nAfter SMOTE:")
print(y_train.value_counts())

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

print("Model Trained Successfully!")

predictions = model.predict(X_test)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# ==========================
# FEATURE IMPORTANCE
# ==========================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance_df)

# Save Feature Importance CSV
importance_df.to_csv(
    "models/feature_importance.csv",
    index=False
)

# ==========================
# SAVE MODEL & ENCODERS
# ==========================

import joblib

joblib.dump(
    model,
    "models/loan_model.pkl"
)

joblib.dump(
    encoders,
    "models/encoders.pkl"
)

print("\nModel Saved: models/loan_model.pkl")
print("Encoders Saved: models/encoders.pkl")
print("Feature Importance Saved: models/feature_importance.csv")