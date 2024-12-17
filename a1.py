import streamlit as st
import pandas as pd
import base64
from PIL import Image
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

# Function to display the home page
def home_page():
    st.title("AI-powered Smart Agricultural Monitoring System")
    
    st.markdown("""
    ### Beyond basic monitoring, the system leverages AI to offer personalized crop recommendations tailored to the specific conditions of the farm, including:
    
    - **Soil type**
    - **Available investment**
    - **Climatic conditions**
    - **Seasonal factors**

    To maximize the farmer's return on investment, the system provides **strategic advice on the optimal locations and times** to sell the harvested crops, ensuring the highest possible profit for the farmer.
    """)

    st.sidebar.title("About this Application")
    st.sidebar.write("""
    This comprehensive crop recommendation system uses both **investment and soil type** 
    as well as **nutrient, temperature, and pH criteria** to offer the best crop suggestions 
    for maximizing profitability in farming.
    """)
     # Display a button at the bottom of the page
    if st.button("Go"):
        # If the button is clicked, set session state to indicate page change
        st.session_state.page = "next"
    # Custom CSS to position text in the bottom left

    st.markdown(
     """
      <style>
     .bottom-right {
        position: fixed;
        bottom: 0;
       right: 0;
        padding: 10px;
        background-color:blue;
        border-top-left-radius: 10px;
        box-shadow: 0px -2px 5px rgba(0,0,0,0.2);
     }
     </style>
     <div class="bottom-right">
         Developed by:
      Y.Abhiram Chowdary 
      </div>
      """,
    unsafe_allow_html=True
)
# Function to display the next page with crop recommendation system
def next_page():
    # Load the datasets
    crop_data = pd.read_csv('/home/abhiram1289/Desktop/datasets/Crop_Details_all.csv')

    st.title("Comprehensive Crop Recommendation System")

    # Section 1: Investment and Soil Type Based Recommendations
    st.header("Investment and Soil-based Crop Recommendations")

    # Dropdown inputs for investment-based recommendations
    selected_duration = st.selectbox('Select Crop Duration (Months)', sorted(crop_data['Duration (Months)'].unique()))
    selected_investment = st.selectbox('Select Investment per Acre (INR)', sorted(crop_data['Investment per Acre (INR)'].unique()))
    selected_soil_type = st.selectbox('Select Suitable Soil Type', crop_data['Suitable Soil Type'].unique())
    selected_water_requirement = st.selectbox('Select Water Requirement (liters per acre per day)', sorted(crop_data['Water Requirement (liters per acre per day)'].unique()))

    # Filter the crop dataset based on user input
    filtered_crop_data = crop_data[
        (crop_data['Duration (Months)'] <= selected_duration) &
        (crop_data['Investment per Acre (INR)'] <= selected_investment) &
        (crop_data['Suitable Soil Type'].str.contains(selected_soil_type, case=False, na=False)) &
        (crop_data['Water Requirement (liters per acre per day)'] <= selected_water_requirement)
    ]

    # Display crop recommendations based on investment and soil type in a table
    if not filtered_crop_data.empty:
        st.subheader("Recommended Crops Based on Investment, Soil, and Environmental Criteria")
        
        display_columns = ['Crop', 'Best Season to Cultivate', 'Duration (Months)', 'Harvesting Time', 
                           'Investment per Acre (INR)', 'Average Selling Price (INR per Acre)',
                           'Nitrogen Range (kg/acre)', 'Phosphorus Range (kg/acre)', 
                           'Potassium Range (kg/acre)', 'Humidity Level Range (%)', 
                           'Temperature Range (°C)', 'pH Range']
        
        st.table(filtered_crop_data[display_columns])
    else:
        st.write("No crops match the selected investment criteria. Please adjust your inputs.")
        st.stop()

    # Section 2: Nutrient, Humidity, Temperature, and pH Based Recommendations
    st.header("Additional Nutrient, Temperature, and pH-based Crop Recommendations")

    # Dropdown inputs for nutrient and environment-based recommendations
    selected_Nitrogen = st.selectbox('Select Nitrogen Range (kg/acre)', sorted(crop_data['Nitrogen Range (kg/acre)'].unique()))
    selected_Phosphorus = st.selectbox('Select Phosphorus Range (kg/acre)', sorted(crop_data['Phosphorus Range (kg/acre)'].unique()))
    selected_Potassium = st.selectbox('Select Potassium Range (kg/acre)', sorted(crop_data['Potassium Range (kg/acre)'].unique()))
    selected_humidity = st.selectbox('Select Humidity Level (%)', sorted(crop_data['Humidity Level Range (%)'].unique()))
    selected_temperature = st.selectbox('Select Temperature Range (°C)', sorted(crop_data['Temperature Range (°C)'].unique()))
    selected_pH_Range = st.selectbox('Select pH Range', sorted(crop_data['pH Range'].unique()))

    # Filter the nutrient dataset based on user input
    filtered_nutrient_data = crop_data[
        (crop_data['Nitrogen Range (kg/acre)'] == selected_Nitrogen) &
        (crop_data['Phosphorus Range (kg/acre)'] == selected_Phosphorus) &
        (crop_data['Potassium Range (kg/acre)'] == selected_Potassium) &
        (crop_data['Humidity Level Range (%)'] == selected_humidity) &
        (crop_data['Temperature Range (°C)'] == selected_temperature) &
        (crop_data['pH Range'] == selected_pH_Range)
    ]

    # Display crop recommendations based on nutrients and environmental factors
    if not filtered_nutrient_data.empty:
        st.subheader("Recommended Crops Based on Environmental Criteria")
        display_columns = ['Crop', 'Nitrogen Range (kg/acre)', 'Phosphorus Range (kg/acre)', 
                           'Potassium Range (kg/acre)', 'Humidity Level Range (%)', 
                           'Temperature Range (°C)', 'pH Range']
        
        st.table(filtered_nutrient_data[display_columns])
    else:
        st.write("No crops match the environmental criteria. Please adjust your inputs.")

    # Compare recommendations from both sections
    st.header("Final Recommendation")
    common_recommendations = set(filtered_crop_data['Crop']).intersection(filtered_nutrient_data['Crop'])
    
    if common_recommendations:
        st.subheader("Final Recommended Crops")
        for crop in common_recommendations:
            st.write(f"**Crop**: {crop}")
    else:
        st.subheader("No common crops found between investment/soil and environmental/nutrient-based recommendations.")
        st.write("You may need to adjust your inputs or consult fertilizer suggestions based on the crops recommended in each section.")
    st.sidebar.title("crop growing suggestions ")
    st.sidebar.header("we can monitor your crops")
    st.sidebar.write("""
        you can upload the images inthe below of your crops we suggest fertilizers to kill the diseases and to increase the crop yeild 
         """)
    
# File uploader in the sidebar
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
# If an image is uploaded
    if uploaded_file is not None:
    # Open the uploaded image
      image = Image.open(uploaded_file)
     
    # Display the image in the main section
      st.sidebar.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Optionally, display some details
      st.sidebar.write(f"Image format: {image.format}")
      st.sidebar.write(f"Image size: {image.size}")
    else:
      st.sidebar.write("Please upload an image.")

# Set background images for the home page and next page
if 'page' not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    set_page_bg('/home/abhiram1289/Desktop/datasets/aa2.jpeg')  # Home page background
    set_sidebar_bg('/home/abhiram1289/Desktop/datasets/aa1.jpeg')  # Home page sidebar
    home_page()
elif st.session_state.page == "next":
    set_page_bg('/home/abhiram1289/Desktop/datasets/ss2.jpeg')  # Next page background
    set_sidebar_bg('/home/abhiram1289/Desktop/datasets/ss1.jpeg')  # Next page sidebar
    next_page()

# Sidebar information for the next page
#if st.session_state.page == "next":
   

