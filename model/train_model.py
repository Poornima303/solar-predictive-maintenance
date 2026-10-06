import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ==============================
# 1. LOAD DATASET
# ==============================

data = pd.read_csv("dataset/Solar_Power_Generation_data.csv")

print("Original Shape:", data.shape)

# Remove missing values
data = data.dropna().copy()

# Convert datetime
data["datetime_gmt"] = pd.to_datetime(data["datetime_gmt"])

# Sort by time
data = data.sort_values(["gsp_id", "datetime_gmt"])

# ==============================
# 2. CREATE TIME FEATURES
# ==============================

data["hour"] = data["datetime_gmt"].dt.hour
data["day"] = data["datetime_gmt"].dt.day
data["month"] = data["datetime_gmt"].dt.month

# ==============================
# 3. CREATE PREVIOUS-GENERATION FEATURES
# ==============================

data["previous_generation"] = (
    data.groupby("gsp_id")["generation_mw"].shift(1)
)

data["previous_generation_2"] = (
    data.groupby("gsp_id")["generation_mw"].shift(2)
)

data["previous_generation_3"] = (
    data.groupby("gsp_id")["generation_mw"].shift(3)
)

# Remove rows created by shifting
data = data.dropna().copy()

# ==============================
# 4. GENERATION RATIO
# ==============================

data["generation_ratio"] = (
    data["generation_mw"] /
    data["installedcapacity_mwp"]
)

# ==============================
# 5. CREATE ANOMALY TARGET
# ==============================

daytime_data = data[
    (data["hour"] >= 6) &
    (data["hour"] <= 18) &
    (data["generation_mw"] > 0)
]

threshold = daytime_data["generation_ratio"].quantile(0.10)

print("\nAnomaly Threshold:", threshold)

data["maintenance_required"] = (
    (data["hour"] >= 6) &
    (data["hour"] <= 18) &
    (data["generation_mw"] > 0) &
    (data["generation_ratio"] < threshold)
).astype(int)

print("\nMaintenance Target:")
print(data["maintenance_required"].value_counts())

# ==============================
# 6. SELECT FEATURES
# ==============================

features = [
    "gsp_id",
    "previous_generation",
    "previous_generation_2",
    "previous_generation_3",
    "lcl_mw",
    "ucl_mw",
    "installedcapacity_mwp",
    "capacity_mwp",
    "hour",
    "day",
    "month"
]

X = data[features]
y = data["maintenance_required"]

print("\nFeature Shape:", X.shape)

# ==============================
# 7. TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ==============================
# 8. RANDOM FOREST
# ==============================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

print("\nRandom Forest Model trained successfully!")

# ==============================
# 9. PREDICTION
# ==============================

y_pred = model.predict(X_test)

# ==============================
# 10. EVALUATION
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy * 100, "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ==============================
# 11. SAVE MODEL
# ==============================

joblib.dump(model, "model/model.pkl")

print("\nModel saved successfully!")
import matplotlib.pyplot as plt

# Feature Importance
importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Solar Maintenance Prediction - Feature Importance")
plt.gca().invert_yaxis()

plt.tight_layout()
plt.savefig("model/feature_importance.png")

print("\nFeature importance graph saved successfully!")
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Maintenance"],
    yticklabels=["Normal", "Maintenance"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()