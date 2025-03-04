import streamlit as st
import mlflow.pyfunc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page config
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        /* Title Styling */
        .title {
            font-size: 100px;
            font-weight: bold;
            color: #FF4B4B; /* Vibrant red */
            text-align: center;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        /* Predicted Price Styling */
        .predicted-price {
            font-size: 24px;
            font-weight: bold;
            color: #FFFFFF;
            background-color: #28a745; /* Green */
            padding: 10px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        }
        
        /* Sidebar Styling */
        .css-1d391kg {
            background-color: #20232a !important; /* Dark sidebar */
            color: white !important;
        }
        
        /* Button Styling */
        .stButton>button {
            background-color: #FF4B4B !important;
            color: white !important;
            font-size: 18px !important;
            border-radius: 10px !important;
            padding: 10px !important;
            font-weight: bold !important;
        }
    </style>
""", unsafe_allow_html=True)

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")  # Change if needed

# Load the latest registered LinearRegression model from MLflow
@st.cache_resource
def load_latest_model():
    try:
        client = mlflow.tracking.MlflowClient()
        model_name = "LinearRegression"

        # Get the latest version of the model
        latest_versions = client.search_model_versions(f"name='{model_name}'")
        if not latest_versions:
            st.error(f"❌ No model found for {model_name} in MLflow.")
            return None

        latest_version = sorted(latest_versions, key=lambda x: int(x.version), reverse=True)[0]
        model_uri = latest_version.source  # Get the latest model's URI

        st.sidebar.info(f"🔍 Loaded Model: {model_name} (Version {latest_version.version})")
        return mlflow.pyfunc.load_model(model_uri)

    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None

# Load the latest model
model = load_latest_model()

# Load historical stock data
@st.cache_data
def load_historical_data():
    try:
        historical_data = pd.read_csv("Z:\FAANG project\FAANG_DATA_MLFLOW.csv")  # Update with actual path
        historical_data["Date"] = pd.to_datetime(historical_data["Date"])
        historical_data.set_index("Date", inplace=True)

        # Apply log transformation
        X_log_transformed = historical_data[["Open", "High", "Low", "Volume"]].apply(np.log1p)
        y_log_transformed = np.log1p(historical_data["Close"])  # Assuming "Close" is the target variable
        return X_log_transformed, y_log_transformed
    except Exception as e:
        st.error(f"⚠️ Error loading historical data: {e}")
        return None, None

# Load historical data
X_log_transformed, y_log_transformed = load_historical_data()

# Title
st.title("📈 Stock Price Prediction")

st.sidebar.header("📊 User Input Features")

# Input fields
company = st.sidebar.selectbox("🏢 Select a Company", ["Amazon", "Apple", "Facebook", "Google", "Netflix"])
open_val = st.sidebar.number_input("📈 Open Price", format="%.4f")
high_val = st.sidebar.number_input("📊 High Price", format="%.4f")
low_val = st.sidebar.number_input("📉 Low Price", format="%.4f")
volume_val = st.sidebar.number_input("📦 Volume", format="%.4f")

# Create input DataFrame
input_df = pd.DataFrame([[open_val, high_val, low_val, volume_val]], 
                         columns=["Open", "High", "Low", "Volume"])

# Ensure all required company one-hot columns exist
required_columns = ["Company_Amazon", "Company_Apple", "Company_Facebook", "Company_Google", "Company_Netflix"]
for col in required_columns:
    input_df[col] = 0  # Set default value to 0

# Set the selected company to 1
input_df[f"Company_{company}"] = 1

# Log-transform the input data
input_df_log_transformed = input_df.apply(np.log1p)

# Predict button
if st.sidebar.button("🔮 Predict"):
    try:
        # Predict using the model
        prediction_log = model.predict(input_df_log_transformed)
        prediction = np.expm1(prediction_log)  # Reverse log transformation
        
        # Display predicted price with better visibility
        st.markdown(f'<div class="predicted-price">🚀 The predicted stock price is: <b>${prediction[0]:.2f}</b></div>', unsafe_allow_html=True)
        
        # Display historical data visualization
        if X_log_transformed is not None and y_log_transformed is not None:
            st.subheader("📜 Historical Stock Data ")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(x=X_log_transformed.index, y=y_log_transformed, ax=ax)
            ax.set_title(f"📉 Historical Close Prices for {company}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Log Transformed Close Price")
            st.pyplot(fig)
        
            # Display prediction distribution
            st.subheader("📊 Prediction Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(y_log_transformed, kde=True, ax=ax, color="purple")
            ax.axvline(prediction_log[0], color='r', linestyle='--', label=f'🔮 Predicted Price (Log): {prediction_log[0]:.2f}')
            ax.set_title("📈 Distribution of Predicted Stock Prices")
            ax.set_xlabel("Log Transformed Stock Price")
            ax.set_ylabel("Frequency")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("⚠️ Historical stock data is not available. Please check your data source.")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# Sidebar Info
st.sidebar.markdown("""
### 📖 About
This app predicts stock prices using an ML model registered in MLflow.
""")

st.sidebar.markdown("""
### 📝 Instructions
1. 🏢 **Select a Company**.
2. 📈 **Enter stock features** (Open, High, Low, Volume).
3. 🔮 **Click 'Predict'** to get the stock price prediction.
4. 📊 **View historical stock trends** and prediction distribution.
""")
