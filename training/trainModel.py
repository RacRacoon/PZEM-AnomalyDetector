import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# =============================================
# LOAD DATASET
# =============================================
df = pd.read_csv("PZEM/training/kombinasiDataset.csv")

print("Dataset loaded:", df.shape)
print(df.head())

# =============================================
# FEATURE SELECTION (FINAL)
# =============================================
FEATURES = ["power", "powerFactor", "energy"]
TARGET = "label"

X = df[FEATURES]
y = df[TARGET]

# =============================================
# LABEL ENCODING
# =============================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# =============================================
# TRAIN-TEST SPLIT (SEBELUM SCALING)
# =============================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =============================================
# SCALING (FIT HANYA DI TRAIN)
# =============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================
# KNN MODEL
# =============================================
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights="distance",
    metric="euclidean"
)

knn.fit(X_train_scaled, y_train)

# =============================================
# ANOMALY THRESHOLD (BENAR)
# =============================================
distances, _ = knn.kneighbors(X_train_scaled, n_neighbors=2)
real_distances = distances[:, 1]  # tetangga ke-2

# Ambil jarak tetangga ke-2 (hindari jarak nol ke diri sendiri)
real_distances = distances[:, 1]

# Threshold berbasis distribusi
ANOMALY_THRESHOLD = np.percentile(real_distances, 97)

print("Anomaly threshold (97th percentile):", round(ANOMALY_THRESHOLD, 4))


print("Suggested anomaly threshold:", round(ANOMALY_THRESHOLD, 4))

# =============================================
# EVALUATION
# =============================================
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\n=== MODEL PERFORMANCE ===")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =============================================
# SAVE MODEL (FINAL)
# =============================================
joblib.dump(knn, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "labels.pkl")
joblib.dump(ANOMALY_THRESHOLD, "anomaly_threshold.pkl")

print("\nSaved files:")
print("- knn_model.pkl")
print("- scaler.pkl")
print("- labels.pkl")
print("- anomaly_threshold.pkl")
print("FEATURES USED:", FEATURES)
