import pandas as pd
import numpy as np
import wandb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# =============================================
# INIT W&B
# =============================================
wandb.init(
    project="pzem-anomaly-knn",
    name="knn_k5_dataset_v2",
    config={
        "model": "KNN",
        "n_neighbors": 30,
        "weights": "distance",
        "metric": "euclidean",
        "features": ["power", "powerFactor", "energy"],
        "dataset": "kombinasiDataset_v2.csv",
        "test_size": 0.3,
        "anomaly_percentile": 97
    }
)

config = wandb.config

# =============================================
# LOAD DATASET
# =============================================
df = pd.read_csv("training/kombinasiDataset_v2.csv")
wandb.log({"dataset_rows": df.shape[0]})

print("Dataset loaded:", df.shape)

# =============================================
# FEATURE SELECTION
# =============================================
FEATURES = config.features
TARGET = "label"

X = df[FEATURES]
y = df[TARGET]

# =============================================
# LABEL ENCODING
# =============================================
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# =============================================
# TRAIN-TEST SPLIT
# =============================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=config.test_size,
    random_state=42,
    stratify=y_encoded
)

# =============================================
# SCALING
# =============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================
# KNN MODEL
# =============================================
knn = KNeighborsClassifier(
    n_neighbors=config.n_neighbors,
    weights=config.weights,
    metric=config.metric
)

knn.fit(X_train_scaled, y_train)

# =============================================
# ANOMALY THRESHOLD (DISTANCE-BASED)
# =============================================
distances, _ = knn.kneighbors(X_train_scaled, n_neighbors=2)
real_distances = distances[:, 1]

ANOMALY_THRESHOLD = np.percentile(
    real_distances,
    config.anomaly_percentile
)

print("Anomaly threshold:", round(ANOMALY_THRESHOLD, 4))

# Log distribusi jarak
wandb.log({
    "anomaly_threshold": ANOMALY_THRESHOLD,
    "distance_distribution": wandb.Histogram(real_distances)
})

# =============================================
# EVALUATION
# =============================================
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("\nConfusion Matrix:\n", cm)

# Log metrik evaluasi
wandb.log({
    "accuracy": accuracy,
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_test,
        preds=y_pred,
        class_names=label_encoder.classes_
    )
})

# =============================================
# SAVE MODEL
# =============================================
joblib.dump(knn, "knn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoder, "labels.pkl")
joblib.dump(ANOMALY_THRESHOLD, "anomaly_threshold.pkl")

# Log artifacts
artifact = wandb.Artifact(
    name="knn-anomaly-model",
    type="model",
    description="KNN anomaly detection model with distance threshold"
)

artifact.add_file("knn_model.pkl")
artifact.add_file("scaler.pkl")
artifact.add_file("labels.pkl")
artifact.add_file("anomaly_threshold.pkl")

wandb.log_artifact(artifact)

wandb.finish()
