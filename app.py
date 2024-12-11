import numpy as np
import pandas as pd
import streamlit as st
import pandas as pd
import pycountry
import joblib
from datetime import datetime

# Tiêu đề của ứng dụng
st.title("🚗 Dự đoán giá xe ô tô 🚗")

countries = [country.name for country in pycountry.countries]
df = pd.read_csv('filled_data.csv')
car_make_options = df['Make'].dropna().unique().tolist()
car_make_options.sort()

car_body_options = df['Body'].dropna().unique().tolist()
car_body_options.sort()

body_color_options = df['Body color'].dropna().unique().tolist()
body_color_options.sort()

interior_color_options = df['Interior color'].dropna().unique().tolist()
interior_color_options.sort()

interior_material_options = df['Interior material'].dropna().unique().tolist()
interior_material_options.sort()

fuel_options = df['Fuel'].dropna().unique().tolist()
fuel_options.sort()


# Load model
model = joblib.load('model_test_xgboost.pkl')  # Load the trained model (replace with your actual path)

# Helper function for prediction
def predict_price(input_data):
    # Assuming input_data is a DataFrame, you can predict like this
    price_prediction = model.predict(input_data)
    return round(price_prediction[0])

# Tạo form nhập liệu
st.header("Nhập thông tin xe")

col1, col2, col3 = st.columns(3)

# Nhập liệu ở cột đầu tiên
with col1:
    car_make_options.append('Other')
    make = st.selectbox("Make", [""] + car_make_options)
    
    if make == 'Other':
        model_options = ['Other']
    elif make:
        model_options = df[df['Make'] == make]['Model'].dropna().unique().tolist()
        model_options.sort()
        model_options.append('Other')
    else:
        model_options = df['Model'].dropna().unique().tolist()  # Tất cả các model nếu chưa chọn Make
        model_options.sort()
        model_options.append('Other')
        
    # model_options = models_by_make.get(make, [])
    
    model_input = st.selectbox("Model", [""] + model_options)
    
    mileage = st.number_input("Mileage (km) ", min_value=0)
    
    first_registration = st.number_input("Year registration", min_value=2000, max_value=2024, value=2020)
    
    power = st.number_input("Power (hp) **(Value > 0)**", min_value=0)
    
    transmission = st.selectbox("Transmission", ["", "Automatic", "Manual"])     

with col2:
    
    fuel = st.selectbox("Fuel", [""] + fuel_options)
    drive_type = st.selectbox("Drive type", ["", "4x2", "4x4"])
    consumption = st.number_input("Consumption (l/100km) **(Value > 0)**", min_value=0.0)
    co2_emissions = st.number_input("CO2 emissions (g/km) **(Value > 0)**", min_value=0)

    location = st.selectbox("Location", [""] + countries)
      
with col3:
    body_color_options = [color for color in body_color_options if color != "Missing"]
    body_color = st.selectbox("Body color", [""] + body_color_options)
    interior_color_options = [color for color in interior_color_options if color != "Missing"]
    interior_color = st.selectbox("Interior color", [""] + interior_color_options)
    interior_material_options = [material for material in interior_material_options if material != "Missing"]
    interior_material = st.selectbox("Interior material", [""] + interior_material_options)
    body = st.selectbox("Body", [""] + car_body_options)
    doors = st.selectbox("Doors", ["", "4/5 doors", "2/3 doors"])
    seats = st.number_input("Seats", min_value=2, max_value=7, value=5)


## Nút dự đoán và hiển thị dự đoán
button_style = """
<style>
    .center-container {
        display: flex;
        justify-content: center; /* Canh giữa theo chiều ngang */
        align-items: center; /* Canh giữa theo chiều dọc */
        flex-direction: column;
        
    }
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        border: 3px solid #4CAF50;
        border-radius: 12px;
        font-size: 24px; /* Tăng kích thước font */
        padding: 15px 30px; /* Tăng khoảng cách trong nút */
        margin: 15px 0; /* Tăng khoảng cách giữa các nút */
        cursor: pointer;
        text-transform: uppercase;
    }
    div.stButton > button:hover {
        background-color: white;
        color: #4CAF50;
        border: 3px solid #4CAF50;
    }
    .result-box {
        display: flex; 
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 2px solid #3498DB; /* Giảm độ dày viền */
        padding: 10px; /* Giảm padding trong hộp */
        margin-top: 10px; /* Giảm khoảng cách bên ngoài hộp */
        border-radius: 8px; /* Giảm độ bo góc */
        background-color: #EAF2F8;
        box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1); /* Giảm bóng mờ */
    }
    .result-box h3 {
        margin: 3px 0; /* Giảm khoảng cách giữa các dòng */
        font-size: 18px; /* Giảm kích thước chữ tiêu đề */
    }
    .result-box h3.price {
        font-size: 30px; /* Giảm kích thước chữ giá */
    }
</style>
"""


st.markdown(button_style, unsafe_allow_html=True)

# Center container
with st.container():
    st.markdown('<div class="center-container">', unsafe_allow_html=True)

    # Predict button
    if st.button("Dự đoán giá", help="Nhấn để dự đoán giá xe", key="predict_button"):
        # Prepare input data for prediction
        mileage = np.nan if mileage <= 0 else mileage
        power = np.nan if power <= 0 else power
        consumption = np.nan if consumption <= 0 else consumption
        co2_emissions = np.nan if co2_emissions <= 0 else co2_emissions
        make = make if make else np.nan
        model_input = model_input if model_input else np.nan
        location = location if location else np.nan
        transmission = transmission if transmission else np.nan
        fuel = fuel if fuel else np.nan
        drive_type = drive_type if drive_type else np.nan
        body_color = body_color if body_color else np.nan
        interior_color = interior_color if interior_color else np.nan
        interior_material = interior_material if interior_material else np.nan
        body = body if body else np.nan
        doors = doors if doors else np.nan

        input_data = pd.DataFrame({
            'Make': [make],
            'Model': [model_input],
            'Location': [location],
            #'First registration': [first_registration],
            'Mileage': [mileage],
            'Power': [power],
            'Transmission': [transmission],
            'Fuel': [fuel],
            'Drive type': [drive_type],
            'Consumption': [consumption],
            'CO2 emissions': [co2_emissions],
            'Body color': [body_color],
            'Interior color': [interior_color],
            'Interior material': [interior_material],
            'Body': [body],
            'Doors': [doors],
            'Seats': [seats]
        })
        print(input_data)

        # Make the prediction
        predicted_price = predict_price(input_data)

        # Display the predicted price
        st.markdown(
            f"""
            <div class="result-box">
                <h3 style="color: #3498DB; font-size: 20px; font-weight: bold;">
                     Giá xe dự đoán:
                </h3>
                <h3 style="color: #3498DB; font-size: 40px; font-weight: bold;">
                       $ {predicted_price}
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)
         
