import streamlit as st
import pandas as pd
import joblib
import os
import requests
from streamlit_lottie import st_lottie
from PIL import Image
from collections import Counter
from sklearn.ensemble import RandomForestClassifier  # Add any other required models

# ✅ Set page configuration at the very beginning
st.set_page_config(
    page_title="Delta Model Prediction with Accuracy Check",
    page_icon=":bar_chart:",
    layout="wide"
)

# Initialize session state variables
if 'running' not in st.session_state:
    st.session_state.running = False
if 'completed_steps' not in st.session_state:
    st.session_state.completed_steps = []
if 'model_weights' not in st.session_state:
    st.session_state.model_weights = {}

# Load Lottie animation
@st.cache_resource
def load_lottieurl(url: str):
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

# Load machine learning-related image
@st.cache_resource
def load_image(image_path):
    return Image.open(image_path)

# Load machine learning model with error handling
@st.cache_resource
def load_model(model_path):
    try:
        return joblib.load(model_path)
    except ModuleNotFoundError as e:
        st.error(f"Error loading model: {e}. Ensure all dependencies are in requirements.txt.")
        return None

# Lottie animation for visual feedback
lottie_prediction = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_5njp3vgg.json")


# Function to run the prediction process
def run_prediction_process():
    st.markdown("### 🔍 Running Prediction Process")
    uploaded_file = st.session_state.uploaded_file

    if uploaded_file is None:
        st.error("No CSV uploaded yet.")
        return

    # Load the CSV
    test_data = pd.read_csv(uploaded_file)
    st.dataframe(test_data)

    # Clean data
    test_data_cleaned = test_data.dropna(axis=1)
    required_cols = ['Patient_ID', 'LMS', 'Survival', 'PD']
    existing_cols = [col for col in required_cols if col in test_data_cleaned.columns]
    X_test = test_data_cleaned.drop(columns=existing_cols)

    # Load selected features
    features_file_path = os.path.join('Lasso_feature', 'selected_features.pkl')
    selected_features = joblib.load(features_file_path)

    # Validate features
    available_features = list(X_test.columns)
    valid_features = [feature for feature in selected_features if feature in available_features]

    if not valid_features:
        st.error("No valid features found for prediction.")
        return

    X_test_reduced = X_test[valid_features]

    # Prepare DataFrame for results
    combined_results = pd.DataFrame({
        'Patient_ID': test_data_cleaned['Patient_ID'] if 'Patient_ID' in test_data_cleaned.columns else range(len(X_test_reduced)),
        'Actual': test_data_cleaned['LMS'] if 'LMS' in test_data_cleaned.columns else pd.Series([None] * len(X_test_reduced))
    })

    # Model predictions and assign weights
    model_dir = "saved_models"
    model_names = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]

    # User-defined model weights from sidebar
    model_weights = {model.replace("_model.pkl", ""): st.session_state.model_weights[model.replace("_model.pkl", "")] for model in model_names}

    for model_name in model_names:
        model_path = os.path.join(model_dir, model_name)
        model = load_model(model_path)
        if model is None:
            continue  # Skip if the model fails to load
        y_pred = model.predict(X_test_reduced)

        model_label = model_name.replace("_model.pkl", "")
        combined_results[f"{model_label}_Predicted"] = y_pred

    # Weighted Majority Vote Calculation
    st.markdown("### 🏆 Final Weighted Majority Vote Prediction")
    pred_cols = [f"{model_name.replace('_model.pkl', '')}_Predicted" for model_name in model_names]

    def weighted_majority_vote(row):
        vote_counter = Counter()
        for model_col in pred_cols:
            model_label = model_col.replace('_Predicted', '')
            weight = model_weights.get(model_label, 1)
            prediction = row[model_col]
            vote_counter[prediction] += weight
        return vote_counter.most_common(1)[0][0]

    combined_results['Weighted_Majority_Vote'] = combined_results[pred_cols].apply(weighted_majority_vote, axis=1)

    # Add O/X Comparison and Calculate Accuracy
    accuracy_results = {}
    for model_col in pred_cols + ['Weighted_Majority_Vote']:
        result_col = model_col.replace('_Predicted', '') + '_Result'
        combined_results[result_col] = combined_results.apply(lambda row: 'O' if row['Actual'] == row[model_col] else 'X', axis=1)
        correct_predictions = combined_results[result_col].value_counts().get('O', 0)
        total_predictions = len(combined_results)
        accuracy = (correct_predictions / total_predictions) * 100
        accuracy_results[model_col.replace('_Predicted', '')] = round(accuracy, 2)

    # Display results
    st.markdown(f"### 📊 Weighted Majority Vote Accuracy: {accuracy_results['Weighted_Majority_Vote']}%")
    st.dataframe(combined_results)


# Sidebar Controls
def display_sidebar():
    st.sidebar.title("⚙️ Control Panel")
    st.sidebar.write("---")

    # Upload CSV
    uploaded_file = st.sidebar.file_uploader("📁 Upload CSV for Prediction", type="csv")
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    # Set model weights dynamically for models
    model_dir = "saved_models"
    model_names = [f.replace("_model.pkl", "") for f in os.listdir(model_dir) if f.endswith(".pkl")]

    st.sidebar.markdown("### 🎯 Set Model Weights")
    for model in model_names:
        st.session_state.model_weights[model] = st.sidebar.slider(f"{model} Weight", min_value=0.0, max_value=2.0, value=1.0, step=0.1)

    # Run Prediction
    if st.sidebar.button("▶️ Run Prediction"):
        st.session_state.running = True
        run_prediction_process()


# Main App Execution
def main():
    # Display animation
    st_lottie(lottie_prediction, height=200, key="prediction_animation")

    # Main Title
    st.title("🔬 Delta Model Predictions with Weighted Majority Vote and Accuracy Check")

    # Sidebar controls
    display_sidebar()


# Run Streamlit App
if __name__ == "__main__":
    main()
