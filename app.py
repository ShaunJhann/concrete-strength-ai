import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🏗️")

MODEL_FILE = Path("random_forest_concrete_strength_model.pkl")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a page",
    ["Home", "Concrete Strength Predictor", "Dataset and Model", "Program Relevance"]
)

if page == "Home":
    st.title("AI-Based Concrete Compressive Strength Prediction")
    st.subheader("Using Random Forest Regression")
    st.write(
        "This web application predicts concrete compressive strength "
        "using concrete mixture proportions and curing age."
    )
    st.info(
        "This tool is for educational and preliminary estimation only. "
        "It does not replace laboratory testing or professional engineering judgment."
    )

elif page == "Concrete Strength Predictor":
    st.title("Concrete Compressive Strength Predictor")
    st.write("Enter the concrete mixture information below.")

    col1, col2 = st.columns(2)

    with col1:
        cement = st.number_input("Cement (kg/m³)", min_value=0.0, value=300.0)
        blast_furnace_slag = st.number_input(
            "Blast Furnace Slag (kg/m³)", min_value=0.0, value=100.0
        )
        fly_ash = st.number_input("Fly Ash (kg/m³)", min_value=0.0, value=0.0)
        water = st.number_input("Water (kg/m³)", min_value=0.0, value=180.0)

    with col2:
        superplasticizer = st.number_input(
            "Superplasticizer (kg/m³)", min_value=0.0, value=5.0
        )
        coarse_aggregate = st.number_input(
            "Coarse Aggregate (kg/m³)", min_value=0.0, value=1000.0
        )
        fine_aggregate = st.number_input(
            "Fine Aggregate (kg/m³)", min_value=0.0, value=750.0
        )
        age = st.number_input("Age (days)", min_value=1, value=28)

    if st.button("Predict Strength"):
        try:
            model = load_model()

            input_data = pd.DataFrame([{
                "cement": cement,
                "blast_furnace_slag": blast_furnace_slag,
                "fly_ash": fly_ash,
                "water": water,
                "superplasticizer": superplasticizer,
                "coarse_aggregate": coarse_aggregate,
                "fine_aggregate": fine_aggregate,
                "age": age
            }])

            prediction = model.predict(input_data)[0]

          st.success(
    f"Estimated Concrete Compressive Strength: {prediction:.2f} MPa"
)

st.markdown("### Explanation of Result")

st.write(
    f"""
    The predicted concrete compressive strength is {prediction:.2f} MPa.

    This value represents the estimated ability of the concrete mixture
    to withstand compressive loading based on the input materials,
    curing age, and the trained Random Forest Regression model.

    The result is an AI-generated estimate and should be validated
    through laboratory testing before actual engineering use.
    """
)

st.caption(
    "This application is intended for educational and preliminary "
    "estimation purposes only."
)

        except FileNotFoundError:
            st.error(
                "Model file not found. Put the .pkl file in the same folder as app.py."
            )
        except Exception as error:
            st.error(f"Error: {error}")

elif page == "Dataset and Model":
    st.title("Dataset and AI Model")
    st.write(
        "The project uses the UCI Concrete Compressive Strength dataset "
        "with eight input features and compressive strength as the target."
    )
    st.markdown("### AI Algorithm")
    st.write(
        "A Random Forest Regressor is used to predict concrete compressive strength."
    )
    st.markdown("### Evaluation Metrics")
    st.write("The model was evaluated using MAE, RMSE, and R².")
    st.markdown(
        "[UCI Dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)"
    )

elif page == "Program Relevance":
    st.title("Program Relevance")
    st.write(
        "This project is relevant to Civil Engineering because concrete "
        "compressive strength is an important property in construction and structural engineering."
    )
    st.warning(
        "The application is for educational and preliminary estimation only. "
        "Actual concrete decisions require appropriate testing and professional judgment."
    )

st.sidebar.markdown("---")
st.sidebar.caption("Civil Engineering AI Project")
