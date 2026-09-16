import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@st.cache_resource
def load_and_train_model():
    """Load Titanic data and train the model."""
    try:
        import seaborn as sns
        df = sns.load_dataset("titanic")
    except Exception:
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)

    df = df.copy()
    rename_map = {
        "pclass": "Pclass", "sex": "Sex", "age": "Age",
        "sibsp": "SibSp", "parch": "Parch", "fare": "Fare",
        "embarked": "Embarked", "survived": "Survived",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    cols_to_drop = ["PassengerId", "Name", "Ticket", "Cabin", "class", "who", "adult_male", "deck", "embark_town", "alive", "alone"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    for col in ["Age", "Fare"]:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].astype(str).str.lower()

    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
    categorical_features = ["Sex", "Embarked"]

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]), categorical_features),
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
    ])

    model.fit(X, y)
    return model


st.set_page_config(page_title="Titanic Survival Predictor", page_icon="ship", layout="centered")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; text-align: center; color: #1f77b4; }
    .sub-header { text-align: center; color: #666; margin-bottom: 2rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">Titanic Survival Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Would you have survived the Titanic disaster?</p>', unsafe_allow_html=True)

st.divider()

try:
    model = load_and_train_model()
except Exception as exc:
    st.error(f"Error loading model: {exc}")
    st.stop()

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger Class", [1, 2, 3], format_func=lambda x: f"Class {x}")
        sex = st.selectbox("Sex", ["male", "female"])
        age = st.slider("Age", 0, 80, 30)
        sibsp = st.slider("Siblings / Spouses aboard", 0, 10, 0)

    with col2:
        parch = st.slider("Parents / Children aboard", 0, 10, 0)
        fare = st.number_input("Fare", min_value=0.0, max_value=500.0, value=32.0, step=1.0)
        embarked = st.selectbox("Embarked", ["S", "C", "Q"], format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])

    submitted = st.form_submit_button("Predict Survival", use_container_width=True, type="primary")

if submitted:
    user_input = pd.DataFrame([{
        "Pclass": pclass, "Sex": sex, "Age": float(age),
        "SibSp": sibsp, "Parch": parch, "Fare": float(fare), "Embarked": embarked,
    }])

    prediction = model.predict(user_input)[0]
    survival_prob = model.predict_proba(user_input)[0][1]

    st.divider()

    if prediction == 1:
        st.success(f"**Prediction: SURVIVED** — {survival_prob:.1%} survival probability")
    else:
        st.warning(f"**Prediction: DID NOT SURVIVE** — {survival_prob:.1%} survival probability")

    st.progress(survival_prob)

with st.sidebar:
    st.header("About")
    st.info("""
    This app uses a **Random Forest** model trained on the Titanic dataset.

    **Input Features:**
    - Passenger Class (1st, 2nd, 3rd)
    - Age
    - Sex
    - Siblings/Spouses Aboard
    - Parents/Children Aboard
    - Fare
    - Port of Embarkation
    """)
    st.header("Model Info")
    st.write("**Algorithm:** Random Forest (100 trees)")
    st.write("**Max Depth:** 5")
    st.write("**Accuracy:** ~80%")
