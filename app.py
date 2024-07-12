import streamlit as st
import numpy as np
import xgboost as xgb
import pickle

# Load the model
with open("model/xgb_model.pkl", "rb") as model_file:
    xgb_model = pickle.load(model_file)

LABEL = ['Bisa Meminjam (0)', "Tidak Bisa Meminjam (1)"]

# Streamlit app
st.title("Loan Eligibility Prediction")

# Create input fields
Applicant_Age = st.number_input("Applicant Age", min_value=0, max_value=100, value=30)
Work_Experience = st.number_input("Work Experience", min_value=0, max_value=50, value=5)
Marital_Status = st.selectbox("Marital Status", [0, 1])  # 0: Single, 1: Married (you can customize as needed)
House_Ownership = st.selectbox("House Ownership", [0, 1])  # 0: No, 1: Yes (you can customize as needed)
Vehicle_Ownership_Car = st.selectbox("Vehicle Ownership (Car)", [0, 1])  # 0: No, 1: Yes (you can customize as needed)
Occupation = st.number_input("Occupation", min_value=0, max_value=10, value=1)  # Customize as needed
Years_in_Current_Employment = st.number_input("Years in Current Employment", min_value=0, max_value=50, value=10)
Years_in_Current_Residence = st.number_input("Years in Current Residence", min_value=0, max_value=50, value=5)
Annual_Income_IDR = st.number_input("Annual Income (IDR)", min_value=0, value=50000000)

# Prediction button
if st.button("Predict"):
    new_data = np.array([[Applicant_Age, Work_Experience, Marital_Status, House_Ownership, Vehicle_Ownership_Car, Occupation, Years_in_Current_Employment, Years_in_Current_Residence, Annual_Income_IDR]])
    result = xgb_model.predict(new_data)
    result_label = LABEL[int(result[0])]

    st.write("Prediction Result: ", result_label)






# import numpy as np
# import xgboost as xgb
# import pickle
# from flask import Flask
# from flask import render_template
# from flask import request
# app = Flask(__name__)
# # read model
# with open("model/xgb_model.pkl", "rb") as model_file:
#     xgb_model = pickle.load(model_file)
# LABEL = ['Bisa Meminjam (0)', "Tidak Bisa Meminjam (1)"]

# @app.route("/")
# def data():
#         return render_template("index.html")
     
# @app.route("/predict", methods=['POST'])   
# def predict():
#     # getting input with name in HTML form dan ubah dalam bentuk float 
#        Applicant_Age = float(request.form.get("Applicant_Age"))
#        Work_Experience = float(request.form.get("Work_Experience"))
#        Marital_Status = float(request.form.get("Marital_Status"))
#        House_Ownership = float(request.form.get("House_Ownership")) 
#        Vehicle_Ownership_Car = float(request.form.get("Vehicle_Ownership_Car"))
#        Occupation = float(request.form.get("Occupation"))
#        Years_in_Current_Employment = float(request.form.get("Years_in_Current_Employment"))
#        Years_in_Current_Residence = float(request.form.get("Years_in_Current_Residence"))
#        Annual_Income_IDR = float(request.form.get("Annual_Income_IDR"))
#        # Print the text in terminal for verification 
#         # print(sepal_length)
#        new_data = [[Applicant_Age, Work_Experience, Marital_Status, House_Ownership, Vehicle_Ownership_Car, Occupation, Years_in_Current_Employment, Years_in_Current_Residence, Annual_Income_IDR]]
#        result = xgb_model.predict(new_data)
#        result = LABEL[result[0]]
    
#        return render_template("index.html", prediction_result=result, Applicant_Age=Applicant_Age,Work_Experience=Work_Experience,Marital_Status=Marital_Status,House_Ownership=House_Ownership,Vehicle_Ownership_Car=Vehicle_Ownership_Car,Occupation=Occupation,Years_in_Current_Employment=Years_in_Current_Employment,Years_in_Current_Residence=Years_in_Current_Residence,Annual_Income_IDR=Annual_Income_IDR) 
       
# if __name__ == "__main__":
#     app.run(debug=True) 


