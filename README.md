# Customer Churn AI Predictor 🚀

An end-to-end Machine Learning project to predict the likelihood of customer churn. This project consists of a full predictive modeling pipeline and a beautiful, dynamic web application UI built with Flask to serve the model.

## Features ✨
- **Data Pipeline:** Includes a `train.py` script that handles missing values, label-encodes categorical variables, splits data, and trains a highly accurate Random Forest Classifier model.
- **Model Serialization:** Automatically saves the trained model (`model.pkl`), feature order (`features.pkl`), and data categorizations (`encoders.json`) to easily integrate with the web interface without hardcoded values. 
- **Flask Backend:** A lightweight, robust API powered by Flask that dynamically loads the model artifacts and calculates the exact probability of churn based on user input.
- **Modern Web UI:** Features a stylized glassmorphism layout, vibrant gradient backgrounds, responsive data grids, and accessible components. The form fields are generated **dynamically** based on whatever features the model requires!

## Project Structure 📁
- `app.py`: The Flask web application server and API. 
- `train.py`: The machine learning pipeline to clean `customer_churn.csv`, train the Random Forest model, and save `.pkl` / `.json` artifacts.
- `templates/index.html`: The frontend HTML/CSS template hosting the modern predictor interface.
- `requirements.txt`: Frozen library dependencies to ensure entirely reproducible installs. 

## Setup & Installation ⚙️

1. **Clone the repository**
   ```bash
   git clone https://github.com/praveshgrewal/endtoendproejct.git
   cd endtoendproejct
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the Machine Learning Model**
   *Run the script to analyze the CSV, build the `.pkl` models, and save `.json` encoders!*
   ```bash
   python train.py
   ```

5. **Start the Web App**
   ```bash
   python app.py
   ```
   *Navigate to `http://127.0.0.1:5000` in your web browser to access the application.*
