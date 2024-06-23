import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import EuclideanDistance
st.set_page_config(page_title="Insurance Primes Prediction")

@st.cache_resource
def load_model(filepath: str):
    with st.spinner('Loading model...'):
        return pickle.load(open(filepath, 'rb'))

model = load_model('models/knn_model.pkl')

_max_width_(70)
st.title("Prediction des Primes d'Assurance Maladie")

st.markdown("""
##### A web application for predicting Insurance Primes.
""")
st.markdown("**:book: [Repo GitHub](https://github.com/matheuscamposmt/housing_prices_app)**")


st.header("Entrez vos informations")
subcol1, subcol2 = st.columns(2)
with subcol1:
    age = st.number_input("Age")

    sex = st.radio(
        "Sexe",
        ["Masculin", "Feminin"],
        horizontal=True,
    )

    bmi = st.number_input("Indice de masse corporelle", min_value = 18.5, max_value = 24.9)

with subcol2:
    children = st.number_input("Nombre d'Enfants en charge")

    region = st.radio(
        "Région",
        ["Southeast", "Northeast", "Southwest", "Northwest"],
        horizontal=True,
    )

    smoker = st.radio(
        "Fumez-vous !?",
        ["Oui", "Non"],
        horizontal=True,
    )

button = st.button("Prédire", use_container_width=True)


if button:
    with st.spinner('Calculating...'):
        
        if sex == "Masculin":
            sex =  0
        else:
            sex =  1


        if smoker == "Oui":
            smoker = 0
        else:
            smoker = 1

        if region == "Southeast":
            region = 0
        elif region == "Northeast":
            region = 1
        elif region == "Southwest":
            region = 2
        else:
            region = 3

        
        input_data = {
        "age": age,

        "sex" : sex,

        "bmi": bmi,

        "children": children,

        "region": region.lower(),

        "smoker" : smoker
        }
        
        input_df = pd.DataFrame([input_data])

        prediction = model.predict(input_df).squeeze()
        st.session_state['prediction'] = prediction
        st.success("Done!")

if st.session_state['prediction']:
    pred = st.session_state['prediction']
    st.markdown(
        """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 44px;
        font-weight: 700;
        color: green;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    st.metric(label="Prime D'Assurance Maladie", value=f"$ {pred:.2f}")