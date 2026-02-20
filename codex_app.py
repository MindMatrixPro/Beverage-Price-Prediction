import streamlit as st
import pandas as pd
import joblib

# --- Load assets ---
final_model = joblib.load("final_model_version1.pkl")
expected_columns = joblib.load("expected_columns.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# --- Do not touch ---
label_encodings = {
    'age_group': {'18-25': 0, '26-35': 1, '36-45': 2, '46-55': 3, '56-70': 4},
    'income_levels': {'<10L': 1, '10L - 15L': 2, '16L - 25L': 3, '26L - 35L': 4, '> 35L': 5, 'Not Reported': 0},
    'health_concerns': {
        'High (Very health-conscious)': 0,
        'Low (Not very concerned)': 1,
        'Medium (Moderately health-conscious)': 2
    },
    'consume_frequency(weekly)': {'0-2 times': 1, '3-4 times': 2, '5-7 times': 3},
    'preferable_consumption_size': {'Small (250 ml)': 0, 'Large (1L)': 1, 'Medium (500 ml)': 2}
}

feature_options = {
    'age_group': ['18-25', '26-35', '36-45', '46-55', '56-70'],
    'gender': ['F', 'M'],
    'zone': ['Metro', 'Rural', 'Semi-Urban', 'Urban'],
    'occupation': ['Entrepreneur', 'Retired', 'Student', 'Working Professional'],
    'income_levels': ['<10L', '10L - 15L', '16L - 25L', '26L - 35L', '> 35L', 'Not Reported'],
    'health_concerns': ['High (Very health-conscious)', 'Medium (Moderately health-conscious)', 'Low (Not very concerned)'],
    'consume_frequency(weekly)': ['0-2 times', '3-4 times', '5-7 times'],
    'preferable_consumption_size': ['Small (250 ml)', 'Medium (500 ml)', 'Large (1L)'],
    'flavor_preference': ['Exotic', 'Traditional'],
    'typical_consumption_situations': ['Active', 'Casual', 'Social'],
    'current_brand': ['Established', 'Newcomer'],
    'purchase_channel': ['Online', 'Retail Store'],
    'packaging_preference': ['Eco-Friendly', 'Premium', 'Simple'],
    'awareness_of_other_brands': ['0 to 1', '2 to 3', '4+'],
    'reasons_for_choosing_brands': ['Availability', 'Brand Reputation', 'Price', 'Quality']
}

# --- Hidden Scores ---
cf_ab_score = 50.0
zas_score = 50.0
bsi_score = 50.0

# --- Styling ---
st.set_page_config(page_title="Codex AI", layout="wide")

st.markdown("""
    <style>
        .block-container {
            padding-left: 2em;
            padding-right: 2em;
            padding-top: 2em;
            max-width: 100%;
        }
        body {
            background: #fcfcfc;
        }
        h1 {
            text-align: center;
            color: #1d3557;
            font-size: 2.8em;
            margin-top: 0;
            padding: 0.3em 0 0.5em 0;
        }
        .stButton>button {
            background-color: #1d3557;
            color: #f1faee;
            border: none;
            border-radius: 10px;
            font-size: 1em;
            padding: 10px 20px;
        }
        .stSelectbox label {
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1>🥤 AI Beverage Price Estimator</h1>", unsafe_allow_html=True)

# --- UI Form ---
with st.form("ai_beverage_form"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age_group = st.selectbox("🎂 Age Group", feature_options['age_group'])
    with c2:
        gender = st.selectbox("🚻 Gender", feature_options['gender'])
    with c3:
        zone = st.selectbox("🌍 Zone", feature_options['zone'])
    with c4:
        occupation = st.selectbox("💼 Occupation", feature_options['occupation'])

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        income = st.selectbox("💰 Income Level", feature_options['income_levels'])
    with d2:
        freq = st.selectbox("📅 Weekly Consumption", feature_options['consume_frequency(weekly)'])
    with d3:
        brand = st.selectbox("🏷️ Current Brand", feature_options['current_brand'])
    with d4:
        size = st.selectbox("📦 Preferred Size", feature_options['preferable_consumption_size'])

    e1, e2, e3, e4 = st.columns(4)
    with e1:
        awareness = st.selectbox("🧠 Brand Awareness", feature_options['awareness_of_other_brands'])
    with e2:
        reason = st.selectbox("🎯 Reason to Choose", feature_options['reasons_for_choosing_brands'])
    with e3:
        flavor = st.selectbox("🍹 Flavor Preference", feature_options['flavor_preference'])
    with e4:
        channel = st.selectbox("🛒 Purchase Channel", feature_options['purchase_channel'])

    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        packaging = st.selectbox("📦 Packaging Type", feature_options['packaging_preference'])
    with f2:
        situation = st.selectbox("🧊 Consumption Situation", feature_options['typical_consumption_situations'])
    with f3:
        health = st.selectbox("🏃 Health Concerns", feature_options['health_concerns'])

    predict_btn = st.form_submit_button("Predict Beverage Price")

# --- Prediction Logic ---
if predict_btn:
    try:
        raw_input = {
            'age_group': label_encodings['age_group'][age_group],
            'income_levels': label_encodings['income_levels'][income],
            'health_concerns': label_encodings['health_concerns'][health],
            'consume_frequency(weekly)': label_encodings['consume_frequency(weekly)'][freq],
            'preferable_consumption_size': label_encodings['preferable_consumption_size'][size],
            'cf_ab_score': cf_ab_score,
            'zas_score': zas_score,
            'BSI': bsi_score,
            'gender': gender,
            'zone': zone,
            'occupation': occupation,
            'flavor_preference': flavor,
            'typical_consumption_situations': situation,
            'current_brand': brand,
            'purchase_channel': channel,
            'packaging_preference': packaging,
            'awareness_of_other_brands': awareness,
            'reasons_for_choosing_brands': reason
        }

        X_input = pd.DataFrame([raw_input])
        cat_columns = X_input.select_dtypes(include='object').columns.tolist()
        X_input = pd.get_dummies(X_input, columns=cat_columns, drop_first=True)
        X_input = X_input.reindex(columns=expected_columns, fill_value=0)

        y_pred = final_model.predict(X_input)[0]
        prediction = label_encoders.inverse_transform([y_pred])[0]

        st.success(f"💵 Estimated Price Range: {prediction}")

    except Exception as err:
        st.error(f"⚠️ Something went wrong: {err}")
