import streamlit as st
import pickle

# Load models
knn_model = pickle.load(open('knn_model.pkl', 'rb'))
rf_model = pickle.load(open('rf_model.pkl', 'rb'))
svm_model = pickle.load(open('svm_model.pkl', 'rb'))

# Prediction function
def dis_prediction(model, age, sex, pregnant, TT4, T3, T4U, FTI, TSH):
    result = model.predict([[age, sex, pregnant, TT4, T3, T4U, FTI, TSH]])
    return result

# User input
st.title("Thyroid Condition Prediction")
age = st.number_input("Age", min_value=0)  # Add input for Age
sex = st.selectbox("Sex (0 = Male, 1 = Female)", [0, 1])
pregnant = st.selectbox("Pregnant (0 = No, 1 = Yes)", [0, 1])
TT4 = st.number_input("TT4 (Thyroxine level)")
T3 = st.number_input("T3 (Triiodothyronine level)")
T4U = st.number_input("T4U (Thyroxine Uptake)")
FTI = st.number_input("FTI (Free Thyroxine Index)")
TSH = st.number_input("TSH (Thyroid-Stimulating Hormone)")
model_name = st.selectbox("Select Model", ["KNN", "Random Forest", "SVM"])

if st.button("Predict"):
    if model_name == "KNN":
        prediction = dis_prediction(knn_model, age, sex, pregnant, TT4, T3, T4U, FTI, TSH)
    elif model_name == "Random Forest":
        prediction = dis_prediction(rf_model, age, sex, pregnant, TT4, T3, T4U, FTI, TSH)
    elif model_name == "SVM":
        prediction = dis_prediction(svm_model, age, sex, pregnant, TT4, T3, T4U, FTI, TSH)

    class_labels = {0: "Hyperthyroid", 1: "Hypothyroid", 2: "Negative"}
    st.write(f"Prediction: {class_labels[prediction[0]]}")
