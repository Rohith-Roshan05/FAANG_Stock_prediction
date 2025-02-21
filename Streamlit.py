import mlflow
from mlflow.tracking import MlflowClient
import streamlit as st
import mlflow.pyfunc
import pandas as pd

# ✅ Set MLflow Tracking URI (Ensure MLflow server is running)
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# ✅ Fetch the best model from MLflow
def get_best_model():
    try:
        client = MlflowClient()
        registered_models = client.search_registered_models()

        if not registered_models:
            st.error("⚠️ No registered models found.")
            return None, None
        
        # Get the latest version of the first registered model
        best_model_name = registered_models[0].name
        best_model_version = registered_models[0].latest_versions[0].version

        return best_model_name, best_model_version

    except Exception as e:
        st.error(f"❌ Error fetching registered models: {e}")
        return None, None

# ✅ Load the best model dynamically
def load_best_model():
    best_model_name, best_model_version = get_best_model()

    if not best_model_name:
        return None

    # Construct the model URI dynamically
    model_uri = f"models:/{best_model_name}/{best_model_version}"
    
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# ✅ Function to make predictions
def predict_stock_price(model, input_data):
    try:
        prediction = model.predict(input_data)
        return prediction
    except Exception as e:
        st.error(f"❌ Error making prediction: {e}")
        return None

# ✅ Streamlit app
def main():
    st.title("📈 FAANG Stock Market Prediction")

    # Load the best model
    model = load_best_model()
    if not model:
        st.error("⚠️ No model available. Ensure MLflow is running and the model is registered.")
        return

    # ✅ Sidebar Input Form
    st.sidebar.header("Input Features")
    open_price = st.sidebar.number_input("Open Price", value=100.0)
    high_price = st.sidebar.number_input("High Price", value=105.0)
    low_price = st.sidebar.number_input("Low Price", value=95.0)
    close_price = st.sidebar.number_input("Close Price", value=100.0)
    volume = st.sidebar.number_input("Volume", value=1000000)

    # ✅ Categorical Input (Company Selection)
    company = st.sidebar.selectbox("Select Company", ["Amazon", "Apple", "Facebook", "Google", "Netflix"])

    # ✅ One-hot encoding for categorical features
    company_features = {
        "Company_Amazon": 1 if company == "Amazon" else 0,
        "Company_Apple": 1 if company == "Apple" else 0,
        "Company_Facebook": 1 if company == "Facebook" else 0,
        "Company_Google": 1 if company == "Google" else 0,
        "Company_Netflix": 1 if company == "Netflix" else 0,
    }

    # ✅ Ensure feature names match model training
    input_data = pd.DataFrame([{
        'Open': open_price,
        'High': high_price,
        'Low': low_price,
        'Close': close_price,
        'Volume': volume,
        **company_features  # Add categorical features
    }])

    # ✅ Make Prediction
    if st.sidebar.button("Predict"):
        prediction = predict_stock_price(model, input_data)
        if prediction is not None:
            st.success(f"📊 Predicted Stock Price: {prediction[0]:.2f}")

if __name__ == "__main__":
    main()
