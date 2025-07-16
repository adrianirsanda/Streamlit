import pandas as pd
import numpy as np

from catboost import CatBoostRegressor, Pool
import joblib
import streamlit as st

# ============================================

# Load model
model = joblib.load("Model_Catboost.joblib")

# Load data referensi untuk dropdown (dari clean CSV)
df = pd.read_csv("Streamlit_Used_Car.csv")

print(df.columns)


st.title("🚗 Prediksi Harga Mobil Bekas - Saudi Arabia")

# Buat form input
st.header("Masukkan Detail Mobil")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", sorted(df['Make'].dropna().unique()))
    model_name = st.selectbox("Model", sorted(df['Type'].dropna().unique()))
    age = st.selectbox("Usia", sorted(df['Car_Age'].dropna().unique()))
    mileage = st.number_input("Kilometer Tempuh", min_value=0)
    engine_size = st.number_input("Ukuran Mesin (Engine Size dalam liter)", min_value=0.0, step=0.1, format="%.1f")

with col2:
    transmission = st.selectbox("Transmisi", sorted(df['Gear_Type'].dropna().unique()))
    fuel = st.selectbox("Bahan Bakar", sorted(df['Fuel_Type'].dropna().unique()))
    color = st.selectbox("Warna Mobil", sorted(df['Color'].dropna().unique()))
    region = st.selectbox("Wilayah", sorted(df['Region'].dropna().unique()))
    origin = st.selectbox("Asal Mobil", sorted(df['Origin'].dropna().unique()))
    options = st.selectbox("Opsi Mobil", sorted(df['Options'].dropna().unique()))

# Tombol prediksi
if st.button("Prediksi Harga"):
    # Buat dataframe input lengkap
    input_df = pd.DataFrame({
        "Make": [brand],
        "Type": [model_name],
        "Car_Age": [age],
        "Mileage": [mileage],
        "Engine_Size": [engine_size],
        "Gear_Type": [transmission],
        "Fuel_Type": [fuel],
        "Color": [color],
        "Region": [region],
        "Origin": [origin],
        "Options": [options]
    })

    # Ambil urutan kolom dari model
    model_features = model.feature_names_

    # Validasi kolom
    missing_cols = set(model_features) - set(input_df.columns)
    if missing_cols:
        st.error(f"❌ Kolom berikut hilang dari input: {missing_cols}")
    else:
        # Susun ulang kolom agar sesuai dengan model
        input_df = input_df[model_features]

        # Tentukan kolom kategorikal (berdasarkan data kamu)
        cat_features = [
            "Make", "Type", "Gear_Type", "Fuel_Type",
            "Color", "Region", "Origin", "Options"
        ]

        # Buat Pool dan prediksi
        input_pool = Pool(data=input_df, cat_features=cat_features)
        prediction = model.predict(input_pool)[0]

        # Tampilkan hasil
        st.subheader("💰 Hasil Prediksi")
        st.success(f"Perkiraan harga mobil: **SAR {prediction:,.2f}**")
