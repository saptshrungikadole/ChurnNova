import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/Churn.csv")

# -----------------------------
# Data Cleaning
# -----------------------------
# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove missing rows
df.dropna(inplace=True)

# Drop customer ID
df.drop("customerID", axis=1, inplace=True)

# Convert target variable
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# -----------------------------
# Encoding
# -----------------------------
df_encoded = pd.get_dummies(df)

# Split features and target
X = df_encoded.drop("Churn", axis=1)
y = df_encoded["Churn"]

# Save feature names
feature_columns = X.columns

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Train Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# -----------------------------
# Evaluation
# -----------------------------
preds = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, preds)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# -----------------------------
# Save Artifacts
# -----------------------------
pickle.dump(model, open("models/churn_model.pkl", "wb"))
pickle.dump(scaler, open("models/scaler.pkl", "wb"))
pickle.dump(feature_columns, open("models/feature_columns.pkl", "wb"))

print("Model, scaler, and features saved successfully.")