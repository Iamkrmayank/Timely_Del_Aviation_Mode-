import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

def convert_datetime_to_numeric(df, datetime_columns):
    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df[f'{col}_numeric'] = df[col].astype(int) / 10**9
            df.drop(columns=[col], inplace=True)
    return df

def convert_to_numeric(df):
    # Combine date and time columns and convert to numeric
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

    # Handle Delivery Date
    if 'Delivery Date' in df.columns:
        df['Delivery_Datetime'] = pd.to_datetime(df['Delivery Date'])
        df['Delivery_Numeric'] = df['Delivery_Datetime'].astype(int) / 10**9
        df.drop(columns=['Delivery Date', 'Delivery_Datetime'], inplace=True)

    # Convert Sensor Reading to numeric
    if 'Sensor Reading' in df.columns:
        df['Sensor Reading'] = pd.to_numeric(df['Sensor Reading'], errors='coerce')

    return df

st.title('Quality Status Prediction')

st.write("Upload your CSV file:")

uploaded_file = st.file_uploader("Choose a file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    # Convert date and time columns
    datetime_columns = ['Time to Quality Issue ']  # Adjust if needed
    df = convert_datetime_to_numeric(df, datetime_columns)

    df = convert_to_numeric(df)

    # Extract features and target
    X = df.drop(columns=['Quality Status'])
    y = df['Quality Status']

    # Handle categorical columns
    categorical_cols = ['Production Line ID', 'Product Type', 'Operator ID', 'Inspector ID', 'Supplier ID',
                         'Material Type', 'Material Quality Grade', 'Sensor ID', 'Sensor Type',
                         'Quality Issue ID', 'Quality Issue Type']
    
    le = LabelEncoder()
    for col in categorical_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # Split data into training and testing sets
    X = df.drop(columns=['Quality Status'])
    y = df['Quality Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split resampled data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    st.write(f"Accuracy Score: {accuracy:.2f}")
    st.write("Classification Report:")
    st.text(report)

    # Optional: Display confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    st.pyplot(fig)
