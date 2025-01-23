import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
#d
# Set page configuration
st.set_page_config(
    page_title="HEALTH PREDICTOR",
    layout="wide",
    page_icon="🩺",
    initial_sidebar_state="collapsed",
)
import streamlit as st

# Background with rounded corners
sidebar_style = """
<style>
  .sidebar .sidebar-content {
    background-color: #F0F8FF; /* Light Blue color */
    padding: 20px;
    border-top-left-radius: 20px; /* Top left rounded corner */
    border-bottom-left-radius: 20px; /* Bottom left rounded corner */
  }
</style>
"""
st.markdown(sidebar_style, unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    # Set background color and better visual design using CSS (optional)
    # ... (code for sidebar_item_style if desired)

    options = ['Home', 'Diabetes Prediction', 'Heart Disease Prediction',
               'Parkinsons Prediction']
    selected = st.sidebar.selectbox('Health Predictor : Disease Detector', options, index=0)

    # Icon customization using emojis (within Streamlit's capabilities)
    icons = [':house:', ':bar_chart:', ':heart:', ':bust_in_silhouette:', ':file_cabinet:']
    st.sidebar.markdown("  ".join(icons[:len(options)]), unsafe_allow_html=True)  # Slice icons to match options

# Set background color using CSS styling with markdown
# background_color = """
#     <style>
#         body {
#             background-color: #FFFF00; /* Yellow color */
#         }
#     </style>
# """
# st.markdown(background_color, unsafe_allow_html=True)

# # Sidebar for navigation
# with st.sidebar:
#     # Set background color and better visual design using CSS
#     sidebar_style = """
#         <style>
#             .sidebar .sidebar-content {
#                 background-color: #F0F8FF; /* Light Blue color */
#                 padding: 20px;
#                 border-radius: 10px;
#             }
#             .sidebar .sidebar-content .sidebar-item-icon {
#                 border-radius: 50%;
#                 padding: 10px;
#                 background-color: #92A8D1; /* Light Gray color */
#                 margin-right: 10px;
#             }
#             .sidebar .sidebar-content .sidebar-item-label {
#                 font-family: 'Arial', sans-serif; /* Change font family */
#                 font-size: 16px; /* Change font size */
#                 color: #333; /* Change font color */
#             }
#         </style>
#     """
#     st.markdown(sidebar_style, unsafe_allow_html=True)

#     options = ['Home', 'Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction', 'Breast Cancer Prediction']
#     selected = option_menu('Health INFI : Disease Detector',
#                            options,
#                            menu_icon='hospital-fill',
#                            icons=['house-door', 'activity', 'heart', 'person','file-earmark-medical'],
#                            default_index=0)

# Rest of your code...

# getting the working directory of the main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# loading the saved models

diabetes_model = pickle.load(open(f'{working_dir}/Saved Models/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open(f'{working_dir}/Saved Models/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open(f'{working_dir}/Saved Models/parkinsons_model.sav', 'rb'))


# Rest of your code...
# Home Page
if selected == 'Home':
    st.title('Welcome to Health Assistant')
    
    st.markdown("""
    ## Disease Prediction System
    
    Welcome to the Health Disease Prediction System, created by team Aman and Janmejay . 
    This application allows you to predict the likelihood of having different diseases: Diabetes, Heart Disease, Parkinson's Disease.
    The prediction is based on 3 different machine learning models.
    Use the sidebar to navigate to specific prediction pages.

    ### Diabetes Prediction
    This Model utilizes the scikit-learn library to develop a Support Vector Machine (SVM) classifier for predicting diabetes based on the PIMA Diabetes dataset.
    It involves data preprocessing, including standard scaling, and the dataset is split into training and testing sets. A linear kernel SVM model is trained and evaluated for accuracy.
    The script also showcases making predictions on new data and uses the pickle library to save the trained model for future use, enhancing efficiency and reusability in diabetes prediction tasks.

    ### Heart Disease Prediction
    This model employs the scikit-learn library to construct a logistic regression model for predicting heart disease based on a provided dataset.
    The dataset is loaded and examined to understand its structure and content. Essential data preprocessing steps, such as handling missing values and splitting into training and testing sets, are executed.
    Subsequently, the logistic regression model is trained utilizing the training data, and its accuracy is assessed on both the training and test datasets.
    Furthermore, a predictive system is implemented to make predictions on new input data.
    The model concludes by saving the trained model using the pickle library for future deployment in heart disease prediction tasks.

    ### Parkinson's Disease Prediction
    This model utilizes the scikit-learn library to implement a Support Vector Machine (SVM) for predicting Parkinson's disease based on a provided dataset.
    The dataset is loaded and analyzed, including an examination of the first and last 10 rows, checking for missing values, and exploring basic statistics.
    The dataset is then split into training and testing sets, and the features are standardized using StandardScaler. A linear SVM model is trained on the standardized training data, and its accuracy is evaluated on both the training and test datasets.
    Finally, a predictive system is established to make predictions on new input data, and the trained SVM model is saved using the pickle library for future use in Parkinson's disease prediction tasks.
""", unsafe_allow_html=True)
    
    # GitHub icons and link on the left
    st.sidebar.subheader('Connect with Team: ')
    st.sidebar.write("[Aman](https://www.linkedin.com/in/aman-sikarwar-21514a25b/)")
    st.sidebar.write("[Janmejay](link)")


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    # page title
    st.title('Diabetes Prediction using ML')

    # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')


    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    # page title
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'The person is having heart disease'
       
        else:
            heart_diagnosis = 'The person does not have any heart disease'

    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

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

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)

# elif selected == "Breast Cancer Prediction":

#     st.title("Breast Cancer Prediction using ML")

#     # Features for Breast Cancer prediction
#     breast_cancer_features = [
#         'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
#         'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
#         'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
#         'area error', 'smoothness error', 'compactness error', 'concavity error',
#         'concave points error', 'symmetry error', 'fractal dimension error',
#         'worst radius', 'worst texture', 'worst perimeter', 'worst area',
#         'worst smoothness', 'worst compactness', 'worst concavity',
#         'worst concave points', 'worst symmetry', 'worst fractal dimension'
#     ]

#     # Organize input fields in columns
#     col1, col2, col3, col4, col5 = st.columns(5)

#     with col1:
#         for feature in breast_cancer_features[:len(breast_cancer_features)//5]:
#             st.text_input(feature)

#     with col2:
#         for feature in breast_cancer_features[len(breast_cancer_features)//5:2*(len(breast_cancer_features)//5)]:
#             st.text_input(feature)

#     with col3:
#         for feature in breast_cancer_features[2*(len(breast_cancer_features)//5):3*(len(breast_cancer_features)//5)]:
#             st.text_input(feature)

#     with col4:
#         for feature in breast_cancer_features[3*(len(breast_cancer_features)//5):4*(len(breast_cancer_features)//5)]:
#             st.text_input(feature)

#     with col5:
#         for feature in breast_cancer_features[4*(len(breast_cancer_features)//5):]:
#             st.text_input(feature)

#     # Code for Prediction
#     breast_cancer_diagnosis = ''

#     # Creating a button for Prediction
#     if st.button("Breast Cancer Prediction"):
#         user_input = [st.text_input(feature) for feature in breast_cancer_features]

#         user_input = [float(x) if x else 0.0 for x in user_input]

#         breast_cancer_prediction = breast_cancer_model.predict([user_input])

#         if breast_cancer_prediction[0] == 1:
#             breast_cancer_diagnosis = "The tumor is Malignant"
#         else:
#             breast_cancer_diagnosis = "The tumor is Benign"

#     st.success(breast_cancer_diagnosis)
