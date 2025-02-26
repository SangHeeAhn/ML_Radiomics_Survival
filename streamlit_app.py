 import streamlit as st
import pandas as pd
import joblib
import os
import requests
from streamlit_lottie import st_lottie
from PIL import Image
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # 예시 scikit-learn models

# ----------------------------------------------------
# 1) Set Page Config FIRST
# ----------------------------------------------------
st.set_page_config(
    page_title="Delta Model Prediction with Accuracy Check",
    page_icon=":bar_chart:",
    layout="wide"
)

# ----------------------------------------------------
# 2) Initialize Session State
# ----------------------------------------------------
if 'running' not in st.session_state:
    st.session_state.running = False
if 'completed_steps' not in st.session_state:
    st.session_state.completed_steps = []
if 'model_weights' not in st.session_state:
    st.session_state.model_weights = {}

# ----------------------------------------------------
# 3) Cache Resource: Lottie Animations, Images, Models
# ----------------------------------------------------
@st.cache_resource
def load_lottieurl(url: str):
    """Loads a Lottie animation from a given URL."""
    response = requests.get(url)
    if response.status_code != 200:
        return None
    return response.json()

@st.cache_resource
def load_image(image_path):
    """Loads an image (e.g., PNG) using Pillow."""
    return Image.open(image_path)

@st.cache_resource
def load_model(model_path):
    """
    Tries to load a scikit-learn model using joblib.
    Returns None if there's a version mismatch or a missing module.
    """
    try:
        return joblib.load(model_path)
    except (AttributeError, ModuleNotFoundError, KeyError) as e:
        # 여기서 _loss 모듈처럼 내부 경로 에러가 나면 건너뛰도록 함.
        st.warning(f"Skipped loading model '{model_path}' due to error: {e}")
        return None

# ----------------------------------------------------
# 4) Load Lottie Animations (used in UI)
# ----------------------------------------------------
lottie_prediction = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_5njp3vgg.json")

# ----------------------------------------------------
# 5) Main Prediction Process
# ----------------------------------------------------
def run_prediction_process():
    st.markdown("### 🔍 Running Prediction Process")
    uploaded_file = st.session_state.uploaded_file

    if uploaded_file is None:
        st.error("No CSV uploaded. Please upload a file on the left sidebar.")
        return

    # 5a) Load CSV Data
    test_data = pd.read_csv(uploaded_file)
    st.dataframe(test_data)

    # 5b) Drop columns with NaN values
    test_data_cleaned = test_data.dropna(axis=1)

    # 5c) Check for required columns
    required_cols = ['Patient_ID', 'LMS', 'Survival', 'PD']
    existing_cols = [col for col in required_cols if col in test_data_cleaned.columns]
    X_test = test_data_cleaned.drop(columns=existing_cols)

    # 5d) Load Feature Info
    features_file_path = os.path.join('Lasso_feature', 'selected_features.pkl')
    selected_features = joblib.load(features_file_path)

    # 5e) Validate that the CSV has the features we need
    available_features = list(X_test.columns)
    valid_features = [f for f in selected_features if f in available_features]
    if not valid_features:
        st.error("No valid features found in the CSV to match your selected_features.pkl.")
        return

    X_test_reduced = X_test[valid_features]

    # 5f) Prepare results DataFrame
    combined_results = pd.DataFrame({
        'Patient_ID': test_data_cleaned['Patient_ID'] if 'Patient_ID' in test_data_cleaned.columns else range(len(X_test_reduced)),
        'Actual': test_data_cleaned['LMS'] if 'LMS' in test_data_cleaned.columns else pd.Series([None]*len(X_test_reduced))
    })

    # 5g) Load Models and Predict
    model_dir = "saved_models"

    # (추가) saved_models 폴더 상태를 먼저 확인
    try:
        files_in_saved_models = os.listdir(model_dir)
        st.write("Files in 'saved_models' directory:", files_in_saved_models)
    except FileNotFoundError:
        st.error(f"Directory '{model_dir}' does not exist or is not accessible.")
        return

    model_names = [f for f in files_in_saved_models if f.endswith(".pkl")]

    # Store which models loaded successfully vs. failed
    successful_models = []
    failed_models = []

    # Retrieve user-set weights from sidebar
    model_weights = {
        m.replace("_model.pkl", ""): st.session_state.model_weights.get(m.replace("_model.pkl", ""), 1.0)
        for m in model_names
    }

    for model_name in model_names:
        model_path = os.path.join(model_dir, model_name)
        model = load_model(model_path)
        if model is None:
            # Model didn't load properly
            failed_models.append(model_name)
            continue

        # If loaded successfully, do predictions
        y_pred = model.predict(X_test_reduced)
        model_label = model_name.replace("_model.pkl", "")
        combined_results[f"{model_label}_Predicted"] = y_pred
        successful_models.append(model_label)

    # 5h) Notify user if any models failed
    if failed_models:
        st.warning(f"⚠️ Failed to load or predict for models: {', '.join(failed_models)}")

    if not successful_models:
        st.error("No models were successfully loaded. Check your model files / environment.")
        return

    # 5i) Weighted Majority Vote
    st.markdown("### 🏆 Final Weighted Majority Vote Prediction")
    pred_cols = [f"{m}_Predicted" for m in successful_models]

    def weighted_majority_vote(row):
        from collections import Counter
        vote_counter = Counter()
        for col in pred_cols:
            model_label = col.replace('_Predicted', '')
            weight = model_weights.get(model_label, 1.0)
            if col in row:
                prediction = row[col]
                vote_counter[prediction] += weight
        # Return the class with the highest weighted count
        return vote_counter.most_common(1)[0][0] if vote_counter else None

    combined_results['Weighted_Majority_Vote'] = combined_results[pred_cols].apply(weighted_majority_vote, axis=1)

    # 5j) Accuracy + O/X Mark
    accuracy_results = {}
    for col in pred_cols + ['Weighted_Majority_Vote']:
        if col not in combined_results.columns:
            continue
        result_col = col.replace('_Predicted', '') + '_Result'
        combined_results[result_col] = combined_results.apply(
            lambda r: 'O' if r['Actual'] == r[col] else 'X',
            axis=1
        )
        correct_preds = combined_results[result_col].value_counts().get('O', 0)
        total = len(combined_results)
        accuracy = round((correct_preds / total)*100, 2)
        accuracy_results[col.replace('_Predicted', '')] = accuracy

    # 5k) Display final accuracy & results
    st.markdown(f"**Weighted Majority Vote Accuracy**: {accuracy_results.get('Weighted_Majority_Vote', 0)}%")
    st.dataframe(combined_results)

# ----------------------------------------------------
# 6) Sidebar Controls
# ----------------------------------------------------
def display_sidebar():
    st.sidebar.title("⚙️ Control Panel")
    st.sidebar.write("---")

    # 6a) File Uploader for CSV
    uploaded_file = st.sidebar.file_uploader("📁 Upload CSV for Prediction", type="csv")
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    # 6b) Dynamic Model Weights
    model_dir = "saved_models"
    try:
        model_names = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
    except FileNotFoundError:
        st.sidebar.warning(f"Directory '{model_dir}' not found.")
        model_names = []

    models_labels = [m.replace("_model.pkl", "") for m in model_names]

    st.sidebar.markdown("### 🎯 Set Model Weights")
    for label in models_labels:
        if label not in st.session_state.model_weights:
            st.session_state.model_weights[label] = 1.0
        st.session_state.model_weights[label] = st.sidebar.slider(
            f"{label} Weight", min_value=0.0, max_value=2.0,
            value=st.session_state.model_weights[label], step=0.1
        )

    # 6c) Button to run prediction
    if st.sidebar.button("▶️ Run Prediction"):
        st.session_state.running = True
        run_prediction_process()

# ----------------------------------------------------
# 7) Main Execution
# ----------------------------------------------------
def main():
    # 7a) Display Lottie Animation
    st_lottie(lottie_prediction, height=200, key="prediction_animation")

    # 7b) Main Title
    st.title("🔬 Delta Model Predictions with Weighted Majority Vote and Accuracy Check")

    # 7c) Show Sidebar
    display_sidebar()

# ----------------------------------------------------
# 8) Run App
# ----------------------------------------------------
if __name__ == "__main__":
    main()
