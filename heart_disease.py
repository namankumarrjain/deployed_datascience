# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('heart.csv')

# Split the data into features (X) and target (y)
X = df.drop(columns=['target'])  # Assuming 'target' is the column for heart disease presence
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Check the accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


import streamlit as st
import numpy as np

# Load the trained model (from the training code above)
# You might want to save the model and load it here to avoid re-training every time.
# from joblib import dump, load
# model = load('heart_disease_model.joblib')

# Define the app interface using Streamlit
st.title('Heart Disease Prediction')

# Input fields for user data
age = st.number_input('Age', min_value=1, max_value=120, value=25)
sex = st.selectbox('Gender', options=[0, 1], help='0: Female, 1: Male')
cp = st.selectbox('Chest Pain Type', options=[0, 1, 2, 3], help='0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic')
trestbps = st.number_input('Resting Blood Pressure (in mm Hg)', min_value=50, max_value=200, value=120)
chol = st.number_input('Serum Cholestoral (in mg/dL)', min_value=100, max_value=600, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dL', options=[0, 1], help='1 if true, 0 if false')
restecg = st.selectbox('Resting Electrocardiographic Results', options=[0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=50, max_value=250, value=150)
exang = st.selectbox('Exercise Induced Angina', options=[0, 1], help='1 if true, 0 if false')
oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox('Slope of the Peak Exercise ST Segment', options=[0, 1, 2])
ca = st.number_input('Number of Major Vessels (0-3)', min_value=0, max_value=3, value=0)
thal = st.selectbox('Thalassemia', options=[0, 1, 2, 3], help='0: Normal, 1: Fixed Defect, 2: Reversible Defect')

# Predict button
if st.button('Predict'):
    # Prepare the input data for prediction
    input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    input_data = scaler.transform(input_data)  # Ensure the data is scaled

    # Perform prediction
    prediction = model.predict(input_data)

    # Display the result
    if prediction[0] == 1:
        st.write("The model predicts: **Heart Disease Detected**.")
    else:
        st.write("The model predicts: **No Heart Disease Detected**.")
