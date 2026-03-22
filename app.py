from flask import Flask, render_template, request
import pickle
import json
import pandas as pd
import os

app = Flask(__name__)

# Load artifacts
MODEL_FILE = 'model.pkl'
FEATURES_FILE = 'features.pkl'
ENCODERS_FILE = 'encoders.json'

model = None
features = []
encoders = {}

if os.path.exists(MODEL_FILE):
    with open(MODEL_FILE, 'rb') as f:
        model = pickle.load(f)

if os.path.exists(FEATURES_FILE):
    with open(FEATURES_FILE, 'rb') as f:
        features = pickle.load(f)

if os.path.exists(ENCODERS_FILE):
    with open(ENCODERS_FILE, 'r') as f:
        encoders = json.load(f)

@app.route('/')
def home():
    # Pass features and encoders to dynamically generate the form
    return render_template('index.html', features=features, encoders=encoders)

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not features:
        return render_template('index.html', features=features, encoders=encoders, error="Model or features not loaded. Wait for the train script to generate them.")
        
    try:
        input_data = {}
        for feature in features:
            val = request.form.get(feature)
            # Find if feature is numerical or categorical
            if feature in encoders:
                # Categorical: get its integer label index from the classes array
                classes = encoders[feature]
                if val in classes:
                    input_data[feature] = classes.index(val)
                else:
                    # Default
                    input_data[feature] = 0
            else:
                # Numerical
                input_data[feature] = float(val) if val else 0.0
                
        # Create DataFrame from input to predict
        df_input = pd.DataFrame([input_data])
        
        # Ensure column order matches exactly
        df_input = df_input[features]
        
        # Make the prediction
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1] # Probability of Class 1 (churn)
        
        result_text = "Churn Detected" if prediction == 1 else "No Churn"
        return render_template('index.html', 
                               features=features, 
                               encoders=encoders, 
                               prediction=result_text,
                               probability=f"{probability*100:.1f}%")
    except Exception as e:
        return render_template('index.html', features=features, encoders=encoders, error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
