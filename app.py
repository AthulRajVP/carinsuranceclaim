import streamlit as st
import numpy as np
import joblib

# Load the trained model
newmodel = joblib.load("carinsuranceclaim.joblib")

st.title("Car Insurance Claim Prediction")
st.markdown("Enter the customer details below to predict insurance claim:")

# --- User Inputs ---
age = st.number_input("Age", min_value=0, max_value=120, value=30)

gender = st.selectbox("Gender", ["Male", "Female"])
gender_male = 1 if gender == "Male" else 0

policy_type = st.selectbox("Policy Type", ["Gold", "Premium", "Other"])
policy_type_gold = 1 if policy_type == "Gold" else 0
policy_type_premium = 1 if policy_type == "Premium" else 0

vehicle_type = st.selectbox("Vehicle Type", ["SUV", "Sedan", "Truck"])
vehicle_type_suv = 1 if vehicle_type == "SUV" else 0
vehicle_type_sedan = 1 if vehicle_type == "Sedan" else 0
vehicle_type_truck = 1 if vehicle_type == "Truck" else 0

accident_type = st.selectbox("Accident Type", ["Minor", "Total Loss"])
accident_type_minor = 1 if accident_type == "Minor" else 0
accident_type_total_loss = 1 if accident_type == "Total Loss" else 0

annual_premium = st.number_input("Annual Premium", min_value=0.0, value=50000.0)
claim_amount = st.number_input("Claim Amount", min_value=0.0, value=100000.0)

police_report = st.selectbox("Police Report Filed?", ["Yes", "No"])
police_report_yes = 1 if police_report == "Yes" else 0

witness_present = st.selectbox("Witness Present?", ["Yes", "No"])
witness_present_yes = 1 if witness_present == "Yes" else 0

past_claims = st.number_input("Past Claims", min_value=0, value=0)
days_to_claim = st.number_input("Days To Claim", min_value=0, value=10)

incident_location = st.selectbox("Incident Location", ["Urban", "Rural"])
incident_location_urban = 1 if incident_location == "Urban" else 0


# --- Prediction ---
if st.button("Predict"):

    # Rule: Age below 18 cannot claim insurance
    if age < 18:
        st.warning("❌ Customer will not claim insurance (Age below 18).")

    else:
        # Create input array
        input_data = np.array([[age, annual_premium, claim_amount, past_claims, days_to_claim,
                                gender_male, policy_type_gold, policy_type_premium,
                                vehicle_type_suv, vehicle_type_sedan, vehicle_type_truck,
                                accident_type_minor, accident_type_total_loss,
                                police_report_yes, witness_present_yes, incident_location_urban]],
                              dtype=float)

        prediction = newmodel.predict(input_data)

        if prediction[0] == 1:
            st.success("✅ Customer will claim insurance.")
        else:
            st.warning("❌ Customer will not claim insurance.")
    
    
    