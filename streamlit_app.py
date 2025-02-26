import streamlit as st
import pandas as pd
import joblib
import os
import requests
from streamlit_lottie import st_lottie
from PIL import Image
from collections import Counter
from io import StringIO
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # 예시 scikit-learn models

# ----------------------------------------------------
# 1) Set Page Config FIRST
# ----------------------------------------------------
st.set_page_config(
    page_title="Radiomics Delta Model Prediction with Accuracy Check",
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
        st.warning(f"Skipped loading model '{model_path}' due to error: {e}")
        return None

# ----------------------------------------------------
# 4) Load Lottie Animations (used in UI)
# ----------------------------------------------------
lottie_prediction = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_5njp3vgg.json")

# ----------------------------------------------------
# 5) Custom Styles (글자 크기 2단계 증가)
# ----------------------------------------------------
def set_custom_styles():
    st.markdown("""
    <style>
    body {
        background: linear-gradient(to bottom, #f5f5f5, #dfe6e9);
    }
    .block-container {
        background-color: #ffffff;
        padding: 2rem 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .weighted-vote-tab {
        background-color: #fdf2d5;
        padding: 1.5rem;
        border-radius: 10px;
        color: #8a6d3b;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    h1 { font-size: 2.5em !important; }
    h2 { font-size: 2em !important; }
    h3 { font-size: 1.75em !important; }
    h4 { font-size: 1.5em !important; }
    .stButton>button {
        background-color: #0984e3;
        color: #fff;
        border-radius: 10px;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #74b9ff;
        color: #2d3436;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------------------------------------------
# 6) Main Prediction Process
# ----------------------------------------------------
def run_prediction_process():
    st.markdown("### 🔍 Running Prediction Process")
    uploaded_file = st.session_state.uploaded_file

    if uploaded_file is None:
        st.error("No CSV uploaded. Please upload a file on the left sidebar.")
        return

    # 6a) Load CSV Data
    test_data = pd.read_csv(uploaded_file)
    st.dataframe(test_data)

    # 6b) Drop columns with NaN values
    test_data_cleaned = test_data.dropna(axis=1)

    # 6c) Check for required columns and remove them from features
    required_cols = ['Patient_ID', 'LMS', 'Survival', 'PD']
    existing_cols = [col for col in required_cols if col in test_data_cleaned.columns]
    X_test = test_data_cleaned.drop(columns=existing_cols)

    # 6d) Load Feature Info
    features_file_path = os.path.join('Lasso_feature', 'selected_features.pkl')
    selected_features = joblib.load(features_file_path)

    # 6e) Validate that the CSV has the required features
    available_features = list(X_test.columns)
    valid_features = [f for f in selected_features if f in available_features]
    if not valid_features:
        st.error("No valid features found in the CSV to match your selected_features.pkl.")
        return

    X_test_reduced = X_test[valid_features]

    # 6f) Prepare results DataFrame with Patient_ID and Actual outcome
    combined_results = pd.DataFrame({
        'Patient_ID': test_data_cleaned['Patient_ID'] if 'Patient_ID' in test_data_cleaned.columns else range(len(X_test_reduced)),
        'Actual': test_data_cleaned['LMS'] if 'LMS' in test_data_cleaned.columns else pd.Series([None]*len(X_test_reduced))
    })

    # 6g) Load Models and perform predictions
    model_dir = "saved_models"
    try:
        files_in_saved_models = os.listdir(model_dir)
        st.write("Files in 'saved_models' directory:", files_in_saved_models)
    except FileNotFoundError:
        st.error(f"Directory '{model_dir}' does not exist or is not accessible.")
        return

    model_names = [f for f in files_in_saved_models if f.endswith(".pkl")]

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
            failed_models.append(model_name)
            continue

        y_pred = model.predict(X_test_reduced)
        model_label = model_name.replace("_model.pkl", "")
        combined_results[f"{model_label}_Predicted"] = y_pred
        successful_models.append(model_label)

    if failed_models:
        st.warning(f"⚠️ Failed to load or predict for models: {', '.join(failed_models)}")

    if not successful_models:
        st.error("No models were successfully loaded. Check your model files / environment.")
        return

    # 6h) Weighted Majority Vote Calculation
    pred_cols = [f"{m}_Predicted" for m in successful_models]

    def weighted_majority_vote(row):
        vote_counter = Counter()
        for col in pred_cols:
            model_label = col.replace('_Predicted', '')
            weight = model_weights.get(model_label, 1.0)
            if col in row:
                prediction = row[col]
                vote_counter[prediction] += weight
        return vote_counter.most_common(1)[0][0] if vote_counter else None

    combined_results['Weighted_Majority_Vote'] = combined_results[pred_cols].apply(weighted_majority_vote, axis=1)

    # 6i) Prepare summary info for analysis (예: 전체 환자 수)
    total_patients = len(combined_results)
    summary_md = f"**Total Patients:** {total_patients}  \n"

    # 6j) Display results in separate tabs (Weighted Majority Vote 탭을 가장 먼저 표시)
    tab_list = ['Weighted Majority Vote'] + [f"{m} Prediction" for m in successful_models]
    tabs = st.tabs(tab_list)

    for i, tab_name in enumerate(tab_list):
        with tabs[i]:
            if tab_name == 'Weighted Majority Vote':
                st.markdown("<div class='weighted-vote-tab'>", unsafe_allow_html=True)
                st.subheader("🏆 Final Weighted Majority Vote Results")
                st.write(summary_md)
                st.dataframe(combined_results[['Patient_ID', 'Weighted_Majority_Vote'] + pred_cols])
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                model_label = tab_name.replace(" Prediction", "")
                st.subheader(f"🔍 {model_label} Model Results")
                st.dataframe(combined_results[["Patient_ID", f"{model_label}_Predicted"]])

    # 6k) Save results to CSV file (예측된 결과값만 저장: Patient_ID, 각 모델 예측값, Weighted Majority Vote)
    final_columns = ['Patient_ID'] + [f"{m}_Predicted" for m in successful_models] + ['Weighted_Majority_Vote']
    csv_buffer = StringIO()
    combined_results[final_columns].to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    # 별도의 컨테이너에 다운로드 버튼 배치 (결과 화면은 그대로 남음)
    with st.container():
        st.download_button(
            label="⬇️ Download CSV Results",
            data=csv_data,
            file_name="Radiomics_Delta_model_predictions_weighted_vote.csv",
            mime="text/csv"
        )
    st.success("✅ Analysis complete! CSV file is ready for download.")

# ----------------------------------------------------
# 7) Sidebar Controls
# ----------------------------------------------------
def display_sidebar():
    st.sidebar.title("Samsung Medical Center")
    st.sidebar.write("---")

    # 7a) File Uploader for CSV
    uploaded_file = st.sidebar.file_uploader("📁 Upload CSV for Prediction", type="csv")
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file

    # 7b) Dynamic Model Weights
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

    # 7c) Button to run prediction
    if st.sidebar.button("▶️ Run Prediction"):
        st.session_state.running = True
        run_prediction_process()

# ----------------------------------------------------
# 8) Main Execution
# ----------------------------------------------------
def main():
    # Apply custom styles
    set_custom_styles()

    # Display Lottie Animation
    st_lottie(lottie_prediction, height=200, key="prediction_animation")

    # Main Title
    st.title("🔬 Radiomics Delta Model Survival Predictions with Weighted Majority Vote")

    # Sidebar Controls
    display_sidebar()

# ----------------------------------------------------
# 9) Run App
# ----------------------------------------------------
if __name__ == "__main__":
    main()
