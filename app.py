import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Define preprocessing functions
def convert_datetime_to_numeric(df, datetime_columns):
    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df[f'{col}_numeric'] = df[col].astype(int) / 10**9
            df.drop(columns=[col], inplace=True)
    return df

def convert_to_numeric(df):
    date_time_pairs = [
        ('Production Date', 'Production Time'),
        ('Inspection Date', 'Inspection Time'),
        ('Quality Issue Date', 'Quality Issue Time')
    ]
    
    for date_col, time_col in date_time_pairs:
        if date_col in df.columns and time_col in df.columns:
            df[f'{date_col[:-5]}_Datetime'] = pd.to_datetime(df[date_col] + ' ' + df[time_col])
            df[f'{date_col[:-5]}_Numeric'] = df[f'{date_col[:-5]}_Datetime'].astype(int) / 10**9
            df.drop(columns=[date_col, time_col, f'{date_col[:-5]}_Datetime'], inplace=True)
    
    if 'Delivery Date' in df.columns:
        df['Delivery_Datetime'] = pd.to_datetime(df['Delivery Date'])
        df['Delivery_Numeric'] = df['Delivery_Datetime'].astype(int) / 10**9
        df.drop(columns=['Delivery Date', 'Delivery_Datetime'], inplace=True)
    
    if 'Sensor Reading' in df.columns:
        df['Sensor Reading'] = pd.to_numeric(df['Sensor Reading'], errors='coerce')
    
    return df

def preprocess_data(df):
    datetime_columns = ['Time to Quality Issue ']  # Adjust if needed
    df = convert_datetime_to_numeric(df, datetime_columns)
    df = convert_to_numeric(df)
    
    categorical_cols = ['Production Line ID', 'Product Type', 'Operator ID', 'Inspector ID', 'Supplier ID',
                         'Material Type', 'Material Quality Grade', 'Sensor ID', 'Sensor Type',
                         'Quality Issue ID', 'Quality Issue Type']
    
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))
    
    return df

# Streamlit app
def main():
    st.title("Quality Status Prediction")

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # Display the dataframe
        st.write("Uploaded Data:")
        st.write(df.head())
        
        # Preprocess the data
        df = preprocess_data(df)
        
        X = df.drop(columns=['Quality Status'])
        y = df['Quality Status']
        
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)
        
        # Apply SMOTE for class imbalance
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Train the Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf_model.predict(X_test)
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        # Display results
        st.write(f"Accuracy Score: {accuracy:.2f}")
        st.write("Classification Report:")
        st.text(report)

if __name__ == "__main__":
    main()
