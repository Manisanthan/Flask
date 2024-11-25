from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import traceback

# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Load the datasets
try:
    data = pd.read_csv('crop.csv')  # For rainfall and yield (adjust path)
    crop_data = pd.read_csv('limited_top_ten_crops.csv')  # For crop prediction (adjust path)
except FileNotFoundError:
    raise Exception("Dataset files not found. Ensure 'crop.csv' and 'limited_top_ten_crops.csv' are in the project directory.")

# Preprocessing for Rainfall and Yield Prediction
label_encoders = {}
for column in ['Crop', 'Season', 'State']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Features for rainfall and yield predictions
X_rainfall = data.drop(columns=['Annual_Rainfall'])
y_rainfall = data['Annual_Rainfall']

X_yield = data[['Crop_Year', 'State', 'Annual_Rainfall', 'Crop', 'Fertilizer', 'Pesticide']]
y_yield = data['Yield']

# Train RandomForestRegressor models for rainfall and yield
rf_model_rainfall = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_rainfall.fit(X_rainfall, y_rainfall)

rf_model_yield = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_yield.fit(X_yield, y_yield)

# Preprocessing for Crop Prediction
crop_le = LabelEncoder()
crop_data['Crop'] = crop_le.fit_transform(crop_data['Crop'])

# One-hot encode 'Season' and 'State'
crop_data = pd.get_dummies(crop_data, columns=['Season', 'State'], drop_first=True)

# Scale numerical features
scaler = StandardScaler()
crop_data[['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']] = scaler.fit_transform(
    crop_data[['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']]
)

# Features (X) and target (y) for crop prediction
X_crop = crop_data.drop('Crop', axis=1)
y_crop = crop_data['Crop']

# Train RandomForestClassifier for crop prediction
rf_model_crop = RandomForestClassifier(random_state=42)
rf_model_crop.fit(X_crop, y_crop)

# Prediction functions remain unchanged
def predict_rainfall(year, state):
    # Logic for rainfall prediction
    pass

def predict_yield(year, state, crop, rainfall, fertilizer, pesticide):
    # Logic for yield prediction
    pass

def predict_crop(area, rainfall, fertilizer, pesticide, season, state):
    # Logic for crop prediction
    pass

# API routes
@app.route('/')
def home():
    return "Flask API for Crop Prediction is running!"

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Ensure data is provided
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        year = data.get('year')
        state = data.get('place')

        if not year or not state:
            return jsonify({'error': 'Missing "year" or "place" in input data'}), 400

        # Predict Rainfall
        rainfall = predict_rainfall(year, state)

        # Handle yield or crop prediction based on additional inputs
        if all(key in data for key in ['crop', 'fertilizer', 'pesticide']):
            crop = data['crop']
            fertilizer = float(data['fertilizer'])
            pesticide = float(data['pesticide'])
            predicted_yield = predict_yield(year, state, crop, rainfall, fertilizer, pesticide)
            return jsonify({'prediction': predicted_yield, 'type': 'yield'}), 200

        elif all(key in data for key in ['area', 'fertilizer', 'pesticide', 'season']):
            area = float(data['area'])
            fertilizer = float(data['fertilizer'])
            pesticide = float(data['pesticide'])
            season = data['season']
            predicted_crop = predict_crop(area, rainfall, fertilizer, pesticide, season, state)
            return jsonify({'prediction': predicted_crop, 'type': 'crop'}), 200

        # Default to rainfall prediction
        return jsonify({'prediction': rainfall, 'type': 'rainfall'}), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

# Dynamic port for deployment platforms
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)