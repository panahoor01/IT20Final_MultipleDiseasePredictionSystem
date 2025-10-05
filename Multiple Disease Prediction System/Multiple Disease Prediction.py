import pickle
import os
import streamlit as st
from streamlit_option_menu import option_menu

# Use relative paths (works locally and on Streamlit Cloud)
base_path = os.path.join(os.path.dirname(__file__), 'saved models')

# Load the saved models and scalers
diabetes_model = pickle.load(open('C:/Projects/Deploying Machine Learning Model/Multiple Disease Prediction System/saved models/diabetes_model.sav', 'rb'))
diabetes_scaler = pickle.load(open('C:/Projects/Deploying Machine Learning Model/DiabetesPrediction_Folder/diabetes_scaler.sav', 'rb'))

# Load heart disease model and scaler (adjust paths as needed)
heartdisease_model = pickle.load(open('C:/Projects/Deploying Machine Learning Model/Multiple Disease Prediction System/saved models/heartdisease_model.sav', 'rb'))
heart_scaler = pickle.load(open('C:/Projects/Deploying Machine Learning Model/HeartDiseasePrediction_Folder/heartdisease_scaler.sav', 'rb'))  # if you have a separate scaler

# Load Parkinson's model and scaler (adjust paths as needed)
parkinsons_model = pickle.load(open('C:/Projects/Deploying Machine Learning Model/Multiple Disease Prediction System/saved models/parkinsons_model.sav', 'rb'))
parkinsons_scaler = pickle.load(open('C:/Projects/Deploying Machine Learning Model/ParkinsonsDiseaseDetection_Folder/parkinsons_scaler.sav', 'rb'))  # if you have a separate scaler

# Sidebar for Navigation
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('🩺 Diabetes Prediction using Machine Learning')
    
    # Getting the input data from the users
    # Columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure Value')

    with col1:
        SkinThickness = st.text_input('SkinThickness Value')
        
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI Value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    
    with col2:
        Age = st.text_input('Age of the Person')

    diab_diagnosis = ''
    
    if st.button('Get Diabetes Test Result'):
        try:
            # Convert and scale input data
            inputs = [float(Pregnancies), float(Glucose), float(BloodPressure),
                      float(SkinThickness), float(Insulin), float(BMI),
                      float(DiabetesPedigreeFunction), float(Age)]

            scaled_inputs = diabetes_scaler.transform([inputs])
            diab_prediction = diabetes_model.predict(scaled_inputs)

            if diab_prediction[0] == 0:
                diab_diagnosis = '✅ The person is NOT Diabetic'
            else:
                diab_diagnosis = '⚠️ The person IS Diabetic'

        except ValueError:
            st.error("⚠️ Please enter valid numeric values in all fields.")

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('🩺 Heart Disease Prediction using Machine Learning')
    
    # Getting the input data from the users
    # Columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex (1 = male; 0 = female)')
    
    with col3:
        cp = st.text_input('Chest Pain Types (0-3)')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
    
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)')
    
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results (0-2)')
    
    with col2:
        thalach = st.text_input('Maximum Heart Rate Achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina (1 = yes; 0 = no)')
        
    with col1:
        oldpeak = st.text_input('ST Depression induced by Exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment (0-2)')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosapy (0-3)')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
    heartdisease_diagnosis = ''
    
    if st.button('Get Heart Disease Test Result'):
        try:
            # Convert input data - use the CORRECT heart disease variables
            inputs = [float(age), float(sex), float(cp),
                      float(trestbps), float(chol), float(fbs),
                      float(restecg), float(thalach), float(exang),
                      float(oldpeak), float(slope), float(ca), float(thal)]

            # Scale inputs if you trained with scaling, otherwise use inputs directly
            scaled_inputs = heart_scaler.transform([inputs])  # Remove this line if you didn't scale during training
            heartdisease_prediction = heart_disease_model.predict(scaled_inputs)
            
            # Or if no scaling was used:
            # heartdisease_prediction = heart_disease_model.predict([inputs])

            if heartdisease_prediction[0] == 0:
                heartdisease_diagnosis = '✅ The Person\'s Heart is Healthy'
            else:
                heartdisease_diagnosis = '⚠️ The Person has Heart Disease'

        except ValueError:
            st.error("⚠️ Please enter valid numeric values in all fields.")
        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")

    st.success(heartdisease_diagnosis)

# Parkinsons Disease Prediction Page
if selected == 'Parkinsons Prediction':
    st.title('🩺 Parkinsons Prediction using Machine Learning')
    
    # Getting the input data from the users
    # Columns for input fields
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')
        
    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')
    
    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitterperc = st.text_input('MDVP:Jitter(%)')
        
    with col5:
        jitterAbs = st.text_input('MDVP:Jitter(Abs)')
    
    with col1:
        rap = st.text_input('MDVP:RAP')
    
    with col2:
        ppq = st.text_input('MDVP:PPQ')
    
    with col3:
        ddp = st.text_input('Jitter:DDP')
    
    with col4:
        shimmer = st.text_input('MDVP:Shimmer')
        
    with col5:
        shimmerDB = st.text_input('MDVP:Shimmer(dB)')
    
    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')
        
    with col3:
        APQ = st.text_input('MDVP:APQ')
    
    with col4:
        dda = st.text_input('Shimmer:DDA')
    
    with col5:
        NHR = st.text_input('NHR')
    
    with col1:
        HNR = st.text_input('HNR')
        
    with col2:
        RPDE = st.text_input('RPDE')
    
    with col3:
        DFA = st.text_input('DFA')
    
    with col4:
        spread1 = st.text_input('spread1')
        
    with col5:
        spread2 = st.text_input('spread2')
    
    with col1:
        D2 = st.text_input('D2')
        
    with col2:
        PPE = st.text_input('PPE')

    parkinsons_diagnosis = ''
    
    if st.button('Get Parkinsons Disease Test Result'):
        try:
            # Convert input data - use the CORRECT Parkinson's variables
            inputs = [float(fo), float(fhi), float(flo), float(Jitterperc),
                      float(jitterAbs), float(rap), float(ppq), float(ddp),
                      float(shimmer), float(shimmerDB), float(APQ3), float(APQ5),
                      float(APQ), float(dda), float(NHR), float(HNR),
                      float(RPDE), float(DFA), float(spread1), float(spread2),
                      float(D2), float(PPE)]

            # Scale inputs if you trained with scaling, otherwise use inputs directly
            scaled_inputs = parkinsons_scaler.transform([inputs])  # Remove this line if you didn't scale during training
            parkinsons_prediction = parkinsons_model.predict(scaled_inputs)
            
            # Or if no scaling was used:
            # parkinsons_prediction = parkinsons_model.predict([inputs])

            if parkinsons_prediction[0] == 0:
                parkinsons_diagnosis = '✅ The Person does not have Parkinsons Disease'
            else:
                parkinsons_diagnosis = '⚠️ The Person has Parkinsons Disease'

        except ValueError:
            st.error("⚠️ Please enter valid numeric values in all fields.")
        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")

    st.success(parkinsons_diagnosis)
