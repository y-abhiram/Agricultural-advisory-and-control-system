import streamlit as st
import pandas as pd
import base64
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import re

# Function to load the image and convert it to base64
def get_base64(image_file):
    with open(image_file, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode()

# Set the sidebar background image
def set_sidebar_bg(image_path):
    base64_image = get_base64(image_path)
    sidebar_bg_img = f'''
    <style>
    [data-testid="stSidebar"] {{
        background-image: url("data:image/jpeg;base64,{base64_image}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }}
    </style>
    '''
    st.markdown(sidebar_bg_img, unsafe_allow_html=True)

# Set the page background image
def set_page_bg(image_path):
    base64_image = get_base64(image_path)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{base64_image}");
        background-size: cover;
        background-position: top left;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Function to handle string ranges and convert them to numeric averages
def convert_range_to_average(value):
    if isinstance(value, str) and '-' in value:
        numbers = re.findall(r'\d+\.?\d*', value)
        if len(numbers) == 2:
            return (float(numbers[0]) + float(numbers[1])) / 2
    return value

# Function to extract unique soil types from a column with comma-separated values
def get_unique_soil_types(crop_data):
    soil_types = crop_data['Suitable Soil Type'].str.split(',').explode().str.strip().unique()
    return soil_types

# Load and preprocess the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('/home/abhiram1289/Desktop/datasets/Crop_Details_all.csv')
    
    # Apply the conversion to relevant columns
    for column in ['Nitrogen Range (kg/acre)', 'Phosphorus Range (kg/acre)', 'Potassium Range (kg/acre)', 
                   'Humidity Level Range (%)', 'Temperature Range (°C)', 'pH Range']:
        data[column] = data[column].apply(convert_range_to_average)
    
    return data

# Train the machine learning model with preprocessing
@st.cache_data
def train_model(crop_data):
    # Ensure all relevant columns are numeric after preprocessing
    for column in ['Humidity Level Range (%)', 'Temperature Range (°C)', 'pH Range']:
        crop_data[column] = pd.to_numeric(crop_data[column], errors='coerce')

    # Prepare features and target variable
    X = crop_data[['Duration (Months)', 'Investment per Acre (INR)', 'Water Requirement (liters per acre per day)', 
                   'Nitrogen Range (kg/acre)', 'Phosphorus Range (kg/acre)', 'Potassium Range (kg/acre)', 
                   'Humidity Level Range (%)', 'Temperature Range (°C)', 'pH Range', 'Suitable Soil Type']]
    y = crop_data['Crop']

    # One-hot encode the categorical column 'Suitable Soil Type'
    X = pd.get_dummies(X, columns=['Suitable Soil Type'], drop_first=True)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Predict and evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

# Function to display the home page
def home_page():
    st.title("AI-powered Smart Agricultural Monitoring System")

    st.markdown("""
    ### Beyond basic monitoring, the system leverages AI to offer personalized crop recommendations tailored to the specific conditions of the farm, including:
    
    - **Soil type**
    - **Available investment**
    - **Climatic conditions**
    - **Seasonal factors**
    """)
    st.sidebar.title("About this Application")
    st.sidebar.write("""
    This comprehensive crop recommendation system uses both **investment and soil type** 
    as well as **nutrient, temperature, and pH criteria** to offer the best crop suggestions 
    for maximizing profitability in farming.
    """)

    if st.button("Go"):
        st.session_state.page = "next"

# Function to display the next page with crop recommendation system
def next_page(crop_data, model, accuracy):
    st.title("Comprehensive Crop Recommendation System")

    st.header("AI-powered Crop Recommendation")

    # Extract unique soil types for the dropdown
    unique_soil_types = get_unique_soil_types(crop_data)

    # User input fields
    selected_duration = st.number_input('Select Crop Duration (Months)', value=4)
    selected_investment = st.number_input('Enter Investment per Acre (INR)', value=25000)
    selected_water_requirement = st.number_input('Water Requirement (liters per acre per day)', value=5000)
    selected_soil_type = st.selectbox('Select Suitable Soil Type', unique_soil_types)

    # Filter crops based on the user's input
    filtered_cropa_data = crop_data[
        (crop_data['Duration (Months)'] <= selected_duration) &
        (crop_data['Investment per Acre (INR)'] <= selected_investment) &
        (crop_data['Suitable Soil Type'].str.contains(selected_soil_type, case=False, na=False)) &
        (crop_data['Water Requirement (liters per acre per day)'] <= selected_water_requirement)
    ]

    # Display crop recommendations based on investment and soil type in a table
    if not filtered_cropa_data.empty:
        st.subheader("Recommended Crops Based on Investment, Soil, and Environmental Criteria")
        
        display_columns = ['Crop', 'Best Season to Cultivate', 'Duration (Months)', 'Harvesting Time', 
                           'Investment per Acre (INR)', 'Average Selling Price (INR per Acre)',
                           'Nitrogen Range (kg/acre)', 'Phosphorus Range (kg/acre)', 
                           'Potassium Range (kg/acre)', 'Humidity Level Range (%)', 
                           'Temperature Range (°C)', 'pH Range']
        
        st.table(filtered_cropa_data[display_columns])
    else:
        st.write("No crops match the selected investment criteria. Please adjust your inputs.")
        st.stop()

    # Additional user input fields for further recommendations
    selected_nitrogen = st.number_input('Nitrogen Range (kg/acre)', value=50)
    selected_phosphorus = st.number_input('Phosphorus Range (kg/acre)', value=30)
    selected_potassium = st.number_input('Potassium Range (kg/acre)', value=40)
    selected_humidity = st.slider('Select Humidity Level (%)', min_value=0, max_value=100, value=50)
    selected_temperature = st.slider('Select Temperature Range (°C)', min_value=0, max_value=50, value=25)
    selected_pH = st.slider('Select pH Range', min_value=0.0, max_value=14.0, value=7.0)

    # Use the model to predict crops
    user_input = pd.DataFrame([[selected_duration, selected_investment, selected_water_requirement, selected_nitrogen,
                                selected_phosphorus, selected_potassium, selected_humidity, selected_temperature, selected_pH, selected_soil_type]],
                              columns=['Duration (Months)', 'Investment per Acre (INR)', 'Water Requirement (liters per acre per day)',
                                       'Nitrogen Range (kg/acre)', 'Phosphorus Range (kg/acre)', 'Potassium Range (kg/acre)',
                                       'Humidity Level Range (%)', 'Temperature Range (°C)', 'pH Range', 'Suitable Soil Type'])

    # One-hot encode the categorical column for prediction
    user_input = pd.get_dummies(user_input, columns=['Suitable Soil Type'], drop_first=True)

    # Match columns to training data
    missing_cols = set(model.feature_names_in_) - set(user_input.columns)
    for col in missing_cols:
        user_input[col] = 0
    user_input = user_input[model.feature_names_in_]

    # Get top 3 crops based on probability
    crop_probabilities = model.predict_proba(user_input)[0]
    top_3_indices = crop_probabilities.argsort()[-12:][::-1]
    top_3_crops = model.classes_[top_3_indices]
    top_3_probs = crop_probabilities[top_3_indices]

    # Display the top 3 recommended crops with their probabilities
    st.subheader("Top 12 Recommended Crops")
    for i, (crop, prob) in enumerate(zip(top_3_crops, top_3_probs)):
        st.write(f"{i+1}. **{crop}** with probability: {prob * 100:.2f}%")

    # Filter the crop_data based on the recommendations and display the table
    filtered_crop_data = crop_data[crop_data['Crop'].isin(top_3_crops)]
    
    st.subheader("Crop Details for the Recommended Crops")
    st.table(filtered_crop_data[display_columns])
    st.header("Final Recommendation")
    common_recommendations = set(filtered_cropa_data['Crop']).intersection(filtered_crop_data['Crop'])
   # st.table(common_recommendations[display_columns])
    if common_recommendations:
        st.subheader("Final Recommended Crops")
        for crop in common_recommendations:
             st.write(f"**Crop**: {crop}")
             
           #  st.table(common_recommendations)
    else:
        st.subheader("No common crops found between investment/soil and environmental/nutrient-based recommendations.")
        st.write("You may need to adjust your inputs or consult fertilizer suggestions based on the crops recommended in each section.")
# Set background images for the home page and next page
if 'page' not in st.session_state:
    st.session_state.page = "home"

# Load data and train the model
crop_data = load_data()
model, accuracy = train_model(crop_data)

if st.session_state.page == "home":
    set_page_bg('/home/abhiram1289/Desktop/datasets/aa2.jpeg')  
    set_sidebar_bg('/home/abhiram1289/Desktop/datasets/aa1.jpeg')  
    home_page()
elif st.session_state.page == "next":
    set_page_bg('/home/abhiram1289/Desktop/datasets/ss2.jpeg')  
    set_sidebar_bg('/home/abhiram1289/Desktop/datasets/ss1.jpeg')  
    next_page(crop_data, model, accuracy)

