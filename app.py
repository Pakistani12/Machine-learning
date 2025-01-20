
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import streamlit as st

# Load the dataset
# Update the path to the absolute file location
diabetes_dataset = pd.read_csv(r'C:\Users\kamra\Desktop\apps using machine learning\diabeties check\diabetes.csv')

# Separate features and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Train the SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Streamlit application
# Set up background image with custom CSS
st.markdown(
    """
    <style>
    body {
        background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRR1RHIUbFbpHAy3Ljo6EV_zy0unc2_virlCw&s"); /* Replace with your image URL */
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.8); /* Adds a transparent white overlay */
        border-radius: 10px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Diabetes Prediction System")
st.write("Enter the values below to check if a person is diabetic:")

# Input fields for user data
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, step=1, format="%d")
glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, step=0.1, format="%.1f")
blood_pressure = st.number_input("Blood Pressure Level", min_value=0.0, max_value=200.0, step=0.1, format="%.1f")
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, step=0.1, format="%.1f")
insulin = st.number_input("Insulin Level", min_value=0.0, max_value=900.0, step=0.1, format="%.1f")
bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, step=0.1, format="%.1f")
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01, format="%.2f")
age = st.number_input("Age", min_value=0, max_value=120, step=1, format="%d")

# Prediction
if st.button("Check Diabetes"):
    # Prepare the input data for prediction
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age])
    input_data_reshaped = input_data.reshape(1, -1)
    
    # Standardize the input data
    std_data = scaler.transform(input_data_reshaped)
    
    # Make prediction
    prediction = classifier.predict(std_data)
    
    # Display the result
    if prediction[0] == 0:
        st.success("The person is not diabetic.")
    else:
        st.error("The person is diabetic.")
