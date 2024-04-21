import streamlit as st
import base64
import base64

#encoded_data = "..."  # Your base64 encoded string
#decoded_data = base64.b64decode(encoded_data).decode('utf-8')  # Decode and handle encoding

import sklearn
import sklearn

import numpy as np
import pickle as pkl
from sklearn.preprocessing import StandardScaler

scal = StandardScaler()

# Load the saved model
# model=pkl.load(open("final_model1.p","rb"))
model = pkl.load(open('trained_model.sav','rb'))

st.set_page_config(page_title="Herat Attack Risk Prediction", page_icon="⚕️", layout="centered",
                   initial_sidebar_state="expanded")
#st.image('logo.png', width=200)


# Predicting the class
def predict_disease(x):
    return model.predict([x])


# Preprocessing user Input
# def preprocess(age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal ):
def preprocess(age, sex, cholesterol , heartrate , diabetes , familyhistory , smoking , obesity, exercisehoursperweek, diet , previousheartproblems,medicationuse ,sedentaryhoursperday, bmi,triglycerides , physicalactivitydaysperweek , sleephoursperday ,BP_Systolic,BP_Diastolic ):
    # Pre-processing user input
    if sex == "male":
        sex = 1
    else:
        sex = 0

    if diabetes == "I've diabetes":
        diabetes = 1
    else:
        diabetes = 0

    if familyhistory  == "I've heart disease family history":
        familyhistory  = 1
    else:
        familyhistory  = 0

    if smoking  == "I smoke":
        smoking  = 1
    else:
        smoking  = 0

    if obesity == "I have obesity":
        obesity = 1
    else:
        obesity = 0

    if diet  == "Average ":
        diet  = 0
    elif diet  == "Healthy":
        diet  = 1
    elif diet  == "Unhealthy":
        diet  = 2

    if previousheartproblems == "Yes":
        previousheartproblems = 1
    else:
        previousheartproblems = 0



    if medicationuse == "Yes":
        medicationuse = 1
    else:
        medicationuse = 0
    


    # if fbs=="Yes":
    #     fbs=1
    # elif fbs=="No":
    #     fbs=0

   #

    # if restecg=="Nothing to note":
    #     restecg=0
    # elif restecg=="ST-T Wave abnormality":
    #     restecg=1
    # elif restecg=="Possible or definite left ventricular hypertrophy":
    #     restecg=2

    # col_names = np.array(['age', 'sex', 'trestbps', 'chol', 'thalach', 'oldpeak', 'cp_1', 'cp_2','cp_3', 'fbs_1', 'restecg_1', 'restecg_2', 'exang_1', 'slope_1','slope_2', 'ca_1', 'ca_2', 'ca_3', 'ca_4', 'thal_1', 'thal_2','thal_3'])
    col_names = np.array(
        ['Age', 'Sex', 'Cholesterol', 'HeartRate', 'Diabetes', 'FamilyHistory', 'Smoking', 'Obesity', 'ExerciseHoursPerWeek', 'Diet', 'PreviousHeartProblems',
         'MedicationUse', 'SedentaryHoursPerDay	', 'BMI', 'Triglycerides', 'PhysicalActivityDaysPerWeek', 'SleepHoursPerDay', 'BP_Systolic','BP_Diastolic'])
    
   

    

    x = np.zeros(len(col_names))
    

    x[0] =age
    x[1] = sex
    x[2] = cholesterol
    x[3] = heartrate
    x[4] = diabetes
    x[5] = familyhistory
    x[6] = smoking
    x[7] = obesity
    x[8] = exercisehoursperweek
    x[9] = diet
    x[10] = previousheartproblems
    x[11] = medicationuse
    x[12] = sedentaryhoursperday
    x[13] = bmi
    x[14] = triglycerides
    x[15] = physicalactivitydaysperweek
    x[16] = sleephoursperday
    x[17] = BP_Systolic
    x[18] = BP_Diastolic
    # 
    from sklearn.preprocessing import StandardScaler 
    scalar = StandardScaler()

    x[0:18] = scalar.fit_transform(x[0:18].reshape(1, -1))

    return x

    # front end elements of the web page


html_temp = """ 
    <div> 
    <h1 style ="color:black;text-align:center;">Heart Attack Risk Prediction </h1> 
     <img src="logo.png" alt=""width:40px;height:40px;"> 

    </div> 
    """

# display the front end aspect
#st.markdown(html_temp, unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 3, 1])  # Adjust column widths as needed
with col2:
    st.image("logo.png", width=500) 
    st.subheader('Heart Attack Risk Prediction')

# following lines create boxes in which user can enter data required to make prediction
age = st.selectbox("Age", range(20, 90, 1))
sex = st.radio("Select Gender: ", ('male', 'female'))
heartrate = st.selectbox('Heart Rate BPM ', range(40, 120, 1))
diabetes = st.radio("Select diabetes state: ", ("I've diabetes", "I don't have diabetes"))
familyhistory  = st.radio("Select family history: ", ("I've heart disease family history", "I don't have heart disease family history"))
smoking  = st.radio("Do you smoke: ", ('I smoke', 'I donot smoke'))
obesity = st.radio("Do you have obesity ", ('I have obesity', 'I donot have obesity'))
# age=st.selectbox ("Age",range(20,90,1))
diet  = st.selectbox('Chest Pain Type', ("Healthy", "Average ", "Unhealthy"))
previousheartproblems = st.radio("Do you had Previous Heart Problems? ", ("Yes", "No"))
medicationuse= st.radio("Do you take midicin? ", ('yes', 'No'))
bmi  = st.number_input('Body mass index')
# restecg=st.selectbox('Resting Electrocardiographic Results',("Nothing to note","ST-T Wave abnormality","Possible or definite left ventricular hypertrophy"))
cholesterol  = st.selectbox('Cholesterol mg/dL ', range(160, 400, 1))

sedentaryhoursperday = st.selectbox('Enter your Sedentary Hours Per Day ', range(0, 12, 1))
sleephoursperday  = st.selectbox('Sleep Hours Per Day', range(1, 11, 1))
exercisehoursperweek  = st.selectbox('Exercise Hours Per Week', range(0, 24, 1))

physicalactivitydaysperweek  = st.selectbox('Physical Activity Days Per Week', range(0, 7, 1))
triglycerides   = st.number_input('Triglycerides value')

BP_Systolic   = st.number_input('Blood pressure Systolic ')
BP_Diastolic   = st.number_input('Blood pressure  Diastolic ')
# user_input=preprocess(sex,cp,exang, fbs, slope, thal )
# pred=preprocess(age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal)


# Basically here we are pre-processing the actual user input
# user_processed_input=preprocess(age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal)
user_processed_input = preprocess(age, sex, cholesterol, heartrate, diabetes, familyhistory, smoking, obesity,exercisehoursperweek, diet, previousheartproblems,medicationuse,sedentaryhoursperday,bmi,triglycerides,physicalactivitydaysperweek,sleephoursperday,BP_Systolic,BP_Systolic)
pred = predict_disease(user_processed_input)

if st.button("Predict"):
    if pred[0] == 0:
        st.success('You have lower risk of getting a heart attack!')

    else:
        st.error('Warning! You have high risk of getting a heart attack!')

st.sidebar.subheader("About App")

st.sidebar.info("This web app is helps you to find out whether you are at a risk of developing a heart attack.")
st.sidebar.info("Enter the required fields and click on the 'Predict' button to check whether you have a healthy heart")
st.sidebar.info("Don't forget to rate this app")


