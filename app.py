import pickle
import pandas as pd
import streamlit as st

# Load the pre-trained model and related objects
with open('rf_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

clf = model_data['model']
scaler = model_data['Scaler']
feature_names = model_data['feature_names']  # Feature names saved during training

# App title
st.title('Stroke Prediction App')

# Define input mappings
gender_mapping = {'Female': 0, 'Male': 1}
married_mapping = {'Yes': 1, 'No': 0}
work_type_mapping = {'Govt_job': 0, 'Private': 1, 'Self-employed': 2, 'children': 3}
residence_mapping = {'Rural': 0, 'Urban': 1}
smoking_status_mapping = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}

# Collect user input
input_data = {
    'gender': gender_mapping[st.selectbox('Choose your gender:', list(gender_mapping.keys()))],
    'age': st.number_input('Input your age', min_value=0, max_value=120, value=25, step=1),
    'hypertension': 1 if st.selectbox('Do you have hypertension?', ['Yes', 'No']) == 'Yes' else 0,
    'heart_disease': 1 if st.selectbox('Do you have heart disease?', ['Yes', 'No']) == 'Yes' else 0,
    'ever_married': married_mapping[st.selectbox('Have you ever married?', list(married_mapping.keys()))],
    'work_type': work_type_mapping[st.selectbox('What is your work type?', list(work_type_mapping.keys()))],
    'Residence_type': residence_mapping[st.selectbox('What is your residence type?', list(residence_mapping.keys()))],
    'avg_glucose_level': st.number_input('Input your average glucose level', min_value=0.0, max_value=300.0, value=100.0, step=0.1),
    'bmi': st.number_input('Input your BMI', min_value=0.0, max_value=100.0, value=25.0, step=0.1),
    'smoking_status': smoking_status_mapping[st.selectbox('What is your smoking status?', list(smoking_status_mapping.keys()))]
}

# Convert input to a DataFrame
df = pd.DataFrame([input_data])

# Ensure all expected features are present (and in the correct order)
for col in feature_names:
    if col not in df.columns:
        df[col] = 0  # Add missing columns with a default value (e.g., 0)

# Reorder columns to match the training order
df = df[feature_names]


if st.button('Predict'):
    try:
        # Validate feature names
        if list(df.columns) != feature_names:
            st.error("Mismatch in feature names or order. Please debug the input data.")
            st.write("Expected feature names:", feature_names)
            st.write("Input feature names:", df.columns.tolist())
            raise ValueError("Feature mismatch detected.")

        # Ensure numerical data is scaled
        numerical = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
        df = scaler.transform(df)

        # Make prediction
        prediction = clf.predict(df)

        # Display result
        if prediction[0] == 1:
            st.write("You are at risk of a stroke. Please consult a healthcare professional.")
        else:
            st.write("You are not at risk of a stroke. Stay healthy!")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
