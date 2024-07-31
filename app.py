#Import libraries
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle



#Import python scripts
#Defining preprocessing from pickel files
def preprocess(features_df):
        lableEncode=pickle.load(open('MultipleLines_Lable_Encoder.pkl','rb'))
        features_df['MultipleLines']=lableEncode.transform(features_df['MultipleLines'])
        lableEncode=pickle.load(open('InternetService_Lable_Encoder.pkl','rb'))
        features_df['InternetService']=lableEncode.transform(features_df['InternetService'])
        lableEncode=pickle.load(open('OnlineSecurity_Lable_Encoder.pkl','rb'))
        features_df['OnlineSecurity']=lableEncode.transform(features_df['OnlineSecurity'])
        lableEncode=pickle.load(open('OnlineBackup_Lable_Encoder.pkl','rb'))
        features_df['OnlineBackup']=lableEncode.transform(features_df['OnlineBackup'])
        lableEncode=pickle.load(open('DeviceProtection_Lable_Encoder.pkl','rb'))
        features_df['DeviceProtection']=lableEncode.transform(features_df['DeviceProtection'])
        lableEncode=pickle.load(open('TechSupport_Lable_Encoder.pkl','rb'))
        features_df['TechSupport']=lableEncode.transform(features_df['TechSupport'])
        lableEncode=pickle.load(open('StreamingTV_Lable_Encoder.pkl','rb'))
        features_df['StreamingTV']=lableEncode.transform(features_df['StreamingTV'])
        lableEncode=pickle.load(open('StreamingMovies_Lable_Encoder.pkl','rb'))
        features_df['StreamingMovies']=lableEncode.transform(features_df['StreamingMovies'])
        lableEncode=pickle.load(open('Contract_Lable_Encoder.pkl','rb'))
        features_df['Contract']=lableEncode.transform(features_df['Contract'])
        lableEncode=pickle.load(open('PaymentMethod_Lable_Encoder.pkl','rb'))
        features_df['PaymentMethod']=lableEncode.transform(features_df['PaymentMethod'])
        std1=pickle.load(open('tenure_standard_scaler.pkl','rb'))
        features_df['tenure'] = std1.transform(features_df[['tenure']])
        std2=pickle.load(open('MonthlyCharges_standard_scaler.pkl','rb'))
        features_df['MonthlyCharges'] = std2.transform(features_df[['MonthlyCharges']])
        std3=pickle.load(open('TotalCharges_standard_scaler.pkl','rb'))
        features_df['TotalCharges'] = std3.transform(features_df[['TotalCharges']])
def main():
    #Setting Application title
    st.title('Telco Customer Churn Prediction App')

      #Setting Application description
    
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    image = Image.open('app.jpg')
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('App to predict Customer Churn')
    st.sidebar.image(image)

    if add_selectbox == "Online":
        yes_no={'Yes':1,'No':0}
        gender_m_f={'Male':1,'Female':0}
        st.info("Input data below")
        #Based on our optimal features selection        
        st.subheader("Demographic data")
        gender = gender_m_f[st.selectbox('Gender:', gender_m_f)]
        seniorcitizen = yes_no[st.selectbox('Senior Citizen:', yes_no)]
        patner = yes_no[st.selectbox('Partner:', yes_no)]
        dependents = yes_no[st.selectbox('Dependent:', yes_no)]
        st.subheader("Payment data")
        tenure = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72, value=0)
        contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
        paperlessbilling = yes_no[st.selectbox('Paperless Billing', yes_no)]
        PaymentMethod = st.selectbox('PaymentMethod',('Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'))
        monthlycharges = st.number_input('The amount charged to the customer monthly', min_value=0, max_value=150)
        totalcharges = st.number_input('The total amount charged to the customer',min_value=0, max_value=10000)

        st.subheader("Services signed up for")
        mutliplelines = st.selectbox("Does the customer have multiple lines",('Yes','No','No phone service'))
        phoneservice = yes_no[st.selectbox('Phone Service:', yes_no)]
        internetservice = st.selectbox("Does the customer have internet service", ('DSL', 'Fiber optic', 'No'))
        onlinesecurity = st.selectbox("Does the customer have online security",('Yes','No','No internet service'))
        onlinebackup = st.selectbox("Does the customer have online backup",('Yes','No','No internet service'))
        deviceProtection = st.selectbox("Does the customer has Device Protection",('Yes','No','No internet service'))
        techsupport = st.selectbox("Does the customer have technology support", ('Yes','No','No internet service'))
        streamingtv = st.selectbox("Does the customer stream TV", ('Yes','No','No internet service'))
        streamingmovies = st.selectbox("Does the customer stream movies", ('Yes','No','No internet service'))

        data = {
                'gender': gender,
                'SeniorCitizen': seniorcitizen,
                'Partner':patner,
                'Dependents': dependents,
                'tenure':tenure,
                'PhoneService': phoneservice,
                'MultipleLines': mutliplelines,
                'InternetService': internetservice,
                'OnlineSecurity': onlinesecurity,
                'OnlineBackup': onlinebackup,
                'DeviceProtection':deviceProtection,
                'TechSupport': techsupport,
                'StreamingTV': streamingtv,
                'StreamingMovies': streamingmovies,
                'Contract': contract,
                'PaperlessBilling': paperlessbilling,
                'PaymentMethod':PaymentMethod,
                'MonthlyCharges': monthlycharges,
                'TotalCharges': totalcharges
                }
        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)
        

        #Calling the Preprocess Function
        preprocess(features_df=features_df)
        

        st.dataframe(features_df)

        #load the model from disk
        model = pickle.load(open('./RnadomForest_model.pkl','rb'))

        prediction = model.predict(features_df)

        if st.button('Predict'):
            if prediction == 1:
                st.warning('Yes, the customer will terminate the service.')
            else:
                st.success('No, the customer is happy with Telco Services.')


    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            features_df = pd.read_csv(uploaded_file)
            features_df.drop('customerID',axis=1,inplace=True)
            features_df.drop('Unnamed: 20',axis=1,inplace=True)
            #Get overview of data
            st.write(features_df.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            #Preprocess inputs
            preprocess(features_df=features_df)
            # Encode categorical features

            #Defining the map function
            def binary_map(feature):
                return feature.map({'Yes':1, 'No':0})

            ## Encoding target feature
            #test_data['Churn'] = test_data[['Churn']].apply(binary_map)

            # Encoding gender category
            features_df['gender'] = features_df['gender'].map({'Male':1, 'Female':0})

            #Encoding other binary category
            binary_list = [ 'SeniorCitizen','Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
            features_df[binary_list] = features_df[binary_list].apply(binary_map)
            if st.button('Predict'):
                #Get batch prediction
                model = pickle.load(open('./RnadomForest_model.pkl','rb'))
                prediction = model.predict(features_df)
                prediction_df = pd.DataFrame(prediction, columns=["Predictions"])
                prediction_df = prediction_df.replace({1:'Yes, the customer will terminate the service.',
                                                    0:'No, the customer is happy with Telco Services.'})

                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(prediction_df)

if __name__ == '__main__':
        main()