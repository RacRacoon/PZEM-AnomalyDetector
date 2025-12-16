from flask import Flask, request, jsonify
import joblib
import pandas as pd
from datetime import datetime
import paho.mqtt.client as mqtt

# ================================
# MQTT CONFIG
# ================================
MQTT_BROKER = "192.168.200.150"
MQTT_PORT = 1883
RELAY_TOPIC = "relay/cut"

CMD_RELAY_ON  = "ON"
CMD_RELAY_CUT = "CUT"

mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

# ================================
# ML COMPONENTS
# ================================
model = joblib.load("knn_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("labels.pkl")

FEATURES = ["power", "powerFactor", "energy"]

# ================================
# ELECTRICAL LIMIT (INDONESIA)
# ================================
V_MIN = 200.0
V_MAX = 240.0
F_MIN = 49.0
F_MAX = 51.0

# ================================
# STATE
# ================================
relay_latched = False

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    global relay_latched

    data = request.json

    power = float(data["power"])
    powerFactor = float(data["powerFactor"])
    energy = float(data["energy"])
    voltage = float(data["voltage"])
    frequency = float(data["frequency"])

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # =====================================================
    # HARD LATCH
    # =====================================================
    if relay_latched:
        print(f"[{ts}] RELAY LATCHED | Electrical fault detected earlier")

        return jsonify({
            "status": "latched",
            "relay": "cut"
        })

    # =====================================================
    # ELECTRICAL SAFETY (ONLY CUTOFF SOURCE)
    # =====================================================
    if not (V_MIN <= voltage <= V_MAX) or not (F_MIN <= frequency <= F_MAX):
        relay_latched = True
        mqtt_client.publish(RELAY_TOPIC, CMD_RELAY_CUT)

        print(
            f"[{ts}] ⚡ ELECTRICAL ANOMALY | "
            f"V={voltage:.1f}V F={frequency:.1f}Hz | RELAY CUT"
        )

        return jsonify({
            "status": "electrical_anomaly",
            "voltage": voltage,
            "frequency": frequency,
            "relay": "cut",
            "latched": True
        })

    # =====================================================
    # IDLE (NO CUTOFF)
    # =====================================================
    if power < 3 and powerFactor < 0.1:
        mqtt_client.publish(RELAY_TOPIC, CMD_RELAY_ON)

        print(f"[{ts}] IDLE | No load connected")

        return jsonify({
            "status": "idle",
            "relay": "on"
        })

    # =====================================================
    # ML INFERENCE (MONITORING ONLY)
    # =====================================================
    X = pd.DataFrame([{
        "power": power,
        "powerFactor": powerFactor,
        "energy": energy
    }], columns=FEATURES)

    X_scaled = scaler.transform(X)

    distances, _ = model.kneighbors(X_scaled, n_neighbors=2)
    distance = float(distances[0][1])

    pred = model.predict(X_scaled)[0]
    label = label_encoder.inverse_transform([pred])[0]
    confidence = model.predict_proba(X_scaled).max()

    if label.lower() == "unknown":
        print(f"[{ts}] UNKNOWN DEVICE | D={distance:.4f}")
    else:
        print(f"[{ts}] NORMAL | {label} | D={distance:.4f}")

    mqtt_client.publish(RELAY_TOPIC, CMD_RELAY_ON)

    return jsonify({
        "device": label,
        "confidence": round(confidence, 3),
        "distance": round(distance, 4),
        "status": "monitoring",
        "relay": "on"
    })


# ================================
# MANUAL RESET
# ================================
@app.route("/reset", methods=["POST"])
def reset():
    global relay_latched

    relay_latched = False
    mqtt_client.publish(RELAY_TOPIC, CMD_RELAY_ON)

    print("MANUAL RESET | RELAY ON")

    return jsonify({
        "status": "reset",
        "relay": "on"
    })


if __name__ == "__main__":
    print("ML Device Monitor + Electrical Protection running on port 5000...")
    app.run(host="0.0.0.0", port=5000)
