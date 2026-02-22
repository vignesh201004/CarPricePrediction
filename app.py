import streamlit as st
import pandas as pd
import joblib
import datetime
import json
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="AutoValue Pro | Smart Valuation", page_icon="🚗", layout="wide")

# --- LOAD ASSETS & METADATA ---
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('car_price_model.pkl')
        columns = joblib.load('model_columns.pkl')
        with open('model_meta.json', 'r') as f:
            meta = json.load(f)
        return model, columns, meta
    except Exception:
        return None, None, None

model, model_columns, meta = load_resources()

# --- SIDEBAR: INPUT CONFIGURATION ---
st.sidebar.title("🛠️ Configuration")
if st.sidebar.button("🔄 Reset System", use_container_width=True):
    st.rerun()

with st.sidebar:
    st.markdown("### Vehicle Details")
    car_name = st.text_input("Brand & Model", "Maruti Swift")
    year = st.slider("Manufacture Year", 2000, datetime.datetime.now().year, 2017)
    present_price = st.number_input("Showroom Price (Full INR)", value=750000, step=50000)
    kms_driven = st.number_input("Total Kms Driven", value=30000, step=5000)
    
    st.markdown("### Technical Specs")
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])
    seller = st.radio("Seller Type", ["Dealer", "Individual"], horizontal=True)
    transmission = st.radio("Transmission", ["Manual", "Automatic"], horizontal=True)

# --- MAIN DASHBOARD ---
st.title("🚗 AutoValue Pro: Indian Market Valuation")
st.write(f"Intelligent Pricing Engine for: **{car_name}**")
st.markdown("---")

if model is None:
    st.error("⚠️ System Error: Model files not found. Run 'python3 car_model.py' first.")
else:
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("🎯 Valuation Result")
        if st.button("Generate Instant INR Valuation", use_container_width=True):
            age = datetime.datetime.now().year - year
            input_data = pd.DataFrame([[present_price, kms_driven, 0, age]], 
                                    columns=['Present_Price', 'Kms_Driven', 'Owner', 'Age'])
            
            for col in model_columns:
                if col not in input_data.columns: input_data[col] = 0
            
            if f"Fuel_Type_{fuel}" in model_columns: input_data[f"Fuel_Type_{fuel}"] = 1
            if f"Seller_Type_{seller}" in model_columns: input_data[f"Seller_Type_{seller}"] = 1
            if f"Transmission_{transmission}" in model_columns: input_data[f"Transmission_{transmission}"] = 1
            
            prediction = max(0, model.predict(input_data[model_columns])[0])
            
            st.balloons()
            st.metric(label="Estimated Resale Value", value=f"₹{prediction:,.2f}")
            st.progress(min(1.0, prediction/present_price), text=f"Value Retention: { (prediction/present_price)*100:.1f}%")

    with col2:
        st.subheader("💡 Market Insights")
        # Dynamic chart based on user's showroom price
        age_range = list(range(0, 16))
        dep_data = pd.DataFrame({
            'Vehicle Age': age_range,
            'Projected Value (₹)': [present_price * (0.88**a) for a in age_range]
        })
        fig = px.area(dep_data, x='Vehicle Age', y='Projected Value (₹)', title="Standard Market Depreciation")
        fig.add_vline(x=datetime.datetime.now().year - year, line_dash="dash", line_color="red", annotation_text="Current Age")
        st.plotly_chart(fig, use_container_width=True)

# --- FOOTER METRICS ---
st.markdown("---")
m1, m2, m3 = st.columns(3)
m1.small("Engine: XGBoost Regressor")
if meta:
    m2.metric("Model Confidence (R²)", f"{meta['accuracy']}%")
    m3.metric("Avg. Error Margin", f"₹{meta['mae']:,}")