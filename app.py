from flask import Flask, render_template, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

import os

model_path = "model/rf_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None
    print("⚠️ Model file not found. Please upload 'rf_model.pkl' to /model directory.")

scaler = joblib.load("model/scaler.pkl")

features = ['PM1', 'PM10', 'NO2', 'O3', 'TEMPERATURE', 'HUMIDITY', 'PRESSURE']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv("data/sample_input.csv")
    X = df[features]
    X_scaled = scaler.transform(X)
    df['Predicted_PM25'] = model.predict(X_scaled)
    return jsonify(df[['Datetime', 'PM2.5', 'Predicted_PM25']].to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

@app.route("/")
def index():
    return "🎉 Flask is running! Model status: " + ("Loaded" if model else "Missing")


