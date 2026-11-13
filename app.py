
import streamlit as st
import pandas as pd
import joblib
# Page Configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load the trained machine learning model
loaded_model = joblib.load("customer_churn_model.pkl")
# Sidebar
st.sidebar.title("Customer Churn")

st.sidebar.write("Machine Learning Project")

st.sidebar.divider()

st.sidebar.subheader("Project Information")

st.sidebar.write("Predict whether a customer may leave the service.")

st.sidebar.subheader("Model Information")

st.sidebar.write("Model: Logistic Regression")

st.sidebar.write("Dataset: Telco Customer Churn")

st.sidebar.divider()

st.sidebar.write("Developed using Python and Streamlit")

# Application title
st.title("Customer Churn Prediction")

st.info(
    "Welcome to our Customer Churn Prediction System! "
    "Enter customer details below to predict the possibility of churn."
)
st.subheader("Enter Customer Details")


# Basic customer details
st.subheader("Basic Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input(
        "Tenure (Months)", min_value=0, max_value=72
    )

    monthly_charges = st.number_input(
        "Monthly Charges", min_value=0.0
    )

with col2:
    total_charges = st.number_input(
        "Total Charges", min_value=0.0
    )


# Personal details
st.subheader("Personal Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox(
        "Gender", ["Female", "Male"]
    )

    senior_citizen = st.selectbox(
        "Senior Citizen", [0, 1]
    )

    partner = st.selectbox(
        "Partner", ["No", "Yes"]
    )

with col2:
    dependents = st.selectbox(
        "Dependents", ["No", "Yes"]
    )

    phone_service = st.selectbox(
        "Phone Service", ["No", "Yes"]
    )



# Contract and Payment Details
st.subheader("Contract and Payment Details")

col1, col2 = st.columns(2)

with col1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
            "Mailed check"
        ]
    )



# Internet and Other Services
st.subheader("Internet and Other Services")

col1, col2 = st.columns(2)

with col1:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["No phone service", "No", "Yes"]
    )

    online_security = st.selectbox(
        "Online Security",
        ["No internet service", "No", "Yes"]
    )

    online_backup = st.selectbox(
        "Online Backup",
        ["No internet service", "No", "Yes"]
    )

    device_protection = st.selectbox(
        "Device Protection",
        ["No internet service", "No", "Yes"]
    )

with col2:
    tech_support = st.selectbox(
        "Tech Support",
        ["No internet service", "No", "Yes"]
    )

    streaming_tv = st.selectbox(
        "Streaming TV",
        ["No internet service", "No", "Yes"]
    )

    streaming_movies = st.selectbox(
        "Streaming Movies",
        ["No internet service", "No", "Yes"]
    )

    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["No", "Yes"]
    )


# Prediction button
if st.button("Predict Churn", type="primary", use_container_width=True):

    # Convert customer details into model features
    customer_data = {
        "SeniorCitizen": senior_citizen,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,

        "gender_Male": int(gender == "Male"),
        "Partner_Yes": int(partner == "Yes"),
        "Dependents_Yes": int(dependents == "Yes"),
        "PhoneService_Yes": int(phone_service == "Yes"),

        "MultipleLines_No phone service": int(
            multiple_lines == "No phone service"
        ),
        "MultipleLines_Yes": int(multiple_lines == "Yes"),

        "InternetService_Fiber optic": int(
            internet_service == "Fiber optic"
        ),
        "InternetService_No": int(internet_service == "No"),

        "OnlineSecurity_No internet service": int(
            online_security == "No internet service"
        ),
        "OnlineSecurity_Yes": int(online_security == "Yes"),

        "OnlineBackup_No internet service": int(
            online_backup == "No internet service"
        ),
        "OnlineBackup_Yes": int(online_backup == "Yes"),

        "DeviceProtection_No internet service": int(
            device_protection == "No internet service"
        ),
        "DeviceProtection_Yes": int(device_protection == "Yes"),

        "TechSupport_No internet service": int(
            tech_support == "No internet service"
        ),
        "TechSupport_Yes": int(tech_support == "Yes"),

        "StreamingTV_No internet service": int(
            streaming_tv == "No internet service"
        ),
        "StreamingTV_Yes": int(streaming_tv == "Yes"),

        "StreamingMovies_No internet service": int(
            streaming_movies == "No internet service"
        ),
        "StreamingMovies_Yes": int(streaming_movies == "Yes"),

        "Contract_One year": int(contract == "One year"),
        "Contract_Two year": int(contract == "Two year"),

        "PaperlessBilling_Yes": int(
            paperless_billing == "Yes"
        ),

        "PaymentMethod_Credit card (automatic)": int(
            payment_method == "Credit card (automatic)"
        ),
        "PaymentMethod_Electronic check": int(
            payment_method == "Electronic check"
        ),
        "PaymentMethod_Mailed check": int(
            payment_method == "Mailed check"
        )
    }

    # Create input DataFrame
    input_df = pd.DataFrame([customer_data])

    # Make prediction
    prediction = loaded_model.predict(input_df)

    # Calculate probabilities
    probability = loaded_model.predict_proba(input_df)

    # Display prediction result
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("Customer may leave!")
    else:
        st.success("Customer may stay!")

    # Display probabilities
    st.write(
        "Stay Probability:",
        round(probability[0][0] * 100, 2), "%"
    )

    st.write(
        "Leave Probability:",
        round(probability[0][1] * 100, 2), "%"
    )
    st.subheader("Probability Analysis")

    st.progress(float(probability[0][0]))

    st.write("Stay Probability")

    st.progress(float(probability[0][1]))

    st.write("Leave Probability")
# Footer
st.divider()

st.markdown(
    "<center>Customer Churn Prediction System</center>",
    unsafe_allow_html=True
)

st.caption("Developed using Python, Machine Learning and Streamlit")