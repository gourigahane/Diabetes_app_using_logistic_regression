import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load the diabetes dataset
df = pd.read_csv('diabetes.csv')

# Handle missing values: replace 0 with NaN in specific columns and impute with median
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)

for col in cols_with_zeros:
    df[col] = df[col].fillna(df[col].median())

# Prepare features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save the model and scaler as pickle files for the app
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the model and scaler exported from Assignment-8_2.ipynb
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

st.title("Diabetes Prediction App")
st.write("Enter patient data to predict diabetes risk using the Logistic Regression model.")

# Create inputs in the exact order established in Assignment-8_2.ipynb
pregnancies = st.number_input("Pregnancies", min_value=0, value=1)
glucose = st.number_input("Glucose", min_value=0, value=120)
bp = st.number_input("Blood Pressure", min_value=0, value=70)
skin = st.number_input("Skin Thickness", min_value=0, value=20)
insulin = st.number_input("Insulin", min_value=0, value=80)
bmi = st.number_input("BMI", min_value=0.0, value=30.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5)
age = st.number_input("Age", min_value=1, value=30)

if st.button("Predict"):
    # Arrange features for scaling and prediction
    features = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    features_scaled = scaler.transform(features)
    
    prediction = model.predict(features_scaled)
    prob = model.predict_proba(features_scaled)[0][1]
    
    if prediction[0] == 1:
        st.error(f"High Risk of Diabetes (Probability: {prob:.2f})")
    else:
        st.success(f"Low Risk of Diabetes (Probability: {prob:.2f})")
