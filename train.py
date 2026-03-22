import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import json

def main():
    # 1. Load the data
    print("Loading data from customer_churn.csv...")
    try:
        df = pd.read_csv("customer_churn.csv")
    except FileNotFoundError:
        print("Error: customer_churn.csv not found.")
        return

    # 2. Data Cleaning
    print("Cleaning data...")
    # Drop CustomerID as it's not a useful feature for prediction
    if 'CustomerID' in df.columns:
        df = df.drop('CustomerID', axis=1)

    # Handle TotalCharges if it's stored as strings with spaces
    if 'TotalCharges' in df.columns:
        # Convert to numeric, setting errors='coerce' turns unparseable values (like spaces) to NaN
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing values
    # For numerical columns, fill with median
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # For categorical columns, fill with mode
    # Explicitly including 'str' avoids Pandas4Warning
    cat_cols = df.select_dtypes(include=['object', 'str']).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])

    # 3. Label Encoding for categorical variables
    print("Encoding categorical variables...")
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # 4. Prepare data for modeling
    if 'Churn' not in df.columns:
        print("Error: 'Churn' column not found in the dataset.")
        return

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Apply Random Forest Classifier
    print("Training Random Forest Classifier...")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # 6. Evaluation
    print("Evaluating model...")
    y_pred = rf_classifier.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 7. Save Model and Artifacts
    print("\nSaving model and artifacts...")
    # Save the model to a pickle file
    with open('model.pkl', 'wb') as f:
        pickle.dump(rf_classifier, f)
    
    # Save the feature columns to a pickle file
    with open('features.pkl', 'wb') as f:
        pickle.dump(list(X.columns), f)
        
    # Save the label encoder classes to a JSON file
    encoders_dict = {col: le.classes_.tolist() for col, le in label_encoders.items()}
    with open('encoders.json', 'w') as f:
        json.dump(encoders_dict, f, indent=4)

    print("Done!")

if __name__ == "__main__":
    main()
