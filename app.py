from flask import Flask, render_template, jsonify
import pandas as pd
import joblib
import os
import urllib.request

app = Flask(__name__)
if not os.path.exists("model"):
    os.makedirs("model")

# Unduh file model dari Dropbox jika belum ada
def download_if_missing(url, path):
    if not os.path.exists(path):
        print(f"📥 Downloading {path}...")
        urllib.request.urlretrieve(url, path)

# Download model dan scaler
download_if_missing("https://www.dropbox.com/scl/fi/hdo9fefydl0wx1z8wxim0/rf_model.pkl?rlkey=w19vp9bwf0xo08jua3uypuqs6&st=3x3syesa&dl=1", "model/rf_model.pkl")
download_if_missing("https://www.dropbox.com/scl/fi/osuagemety2c9kwoiohd5/scaler.pkl?rlkey=h3q3cdhmjp0v0l88uxk5qxnfr&st=ezdxszbq&dl=1", "model/scaler.pkl")

# Load model dan scaler
model = joblib.load("model/rf_model.pkl") if os.path.exists("model/rf_model.pkl") else None
scaler = joblib.load("model/scaler.pkl") if os.path.exists("model/scaler.pkl") else None

# Fitur input
features = ['PM1', 'PM10', 'NO2', 'O3', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE']

# Route root (homepage)
@app.route("/")
def index():
    return "🎉 Flask is running! Model status: " + ("Loaded" if model else "Missing")

# Endpoint prediksi
@app.route("/predict", methods=["POST"])
def predict():
    df = pd.read_csv("data/sample_input.csv")
    X = df[features]
    X_scaled = scaler.transform(X)
    df["Predicted_PM25"] = model.predict(X_scaled)
    return jsonify(df[["Datetime", "PM2.5", "Predicted_PM25"]].to_dict(orient="records"))
except Exception as e:
    return jsonify({"status": "error", "message": str(e)})

# Menjalankan app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
