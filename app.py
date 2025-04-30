import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    st.set_page_config(page_title="Prédiction du Diabète", layout="wide")

    # Titre et objectif
    st.title("🧠 Application de Machine Learning pour la Prédiction du Diabète")
    st.subheader("Auteur : Abdoulaye Diop")
    st.markdown("""
    Cette application a pour objectif de **prédire la probabilité de diabète** chez un patient à partir de variables médicales.
    Elle utilise un modèle de **Random Forest**, entraîné sur un dataset public, pour effectuer les prédictions.
    """)

     # Fonction d'importation des données
    @st.cache_data
    def load_data():
        data = pd.read_csv("diabetes.csv")
        return data


    df = load_data()

    # Sidebar : Paramètres du modèle
    st.sidebar.header("🔧 Paramètres du modèle")
    classifier = st.sidebar.selectbox("Choisir un classificateur", ("Random Forest",))  # SVM, Logistic Regression à venir
    n_arbre = st.sidebar.slider("Nombre d'arbres", 100, 1000, step=10)
    profondeur = st.sidebar.slider("Profondeur max de l'arbre", 1, 20, step=1)
    boots = st.sidebar.radio("Bootstrap ?", ("True", "False")) == "True"

    # Contrôle du nombre de lignes à afficher (sous la sidebar)
    st.sidebar.header("📊 Données")
    n_rows = st.sidebar.number_input("Nombre de lignes à afficher", min_value=1, max_value=len(df), value=5, step=1)

    # Affichage conditionnel du DataFrame
    if st.checkbox("🔍 Afficher les données brutes"):
        st.write(df.head(n_rows))

    # Description des variables
    with st.expander("ℹ️ Description des variables du dataset"):
        st.markdown("""
        - **Pregnancies** : Nombre de grossesses
        - **Glucose** : Concentration de glucose dans le sang (mg/dL)
        - **BloodPressure** : Pression artérielle diastolique (mm Hg)
        - **SkinThickness** : Épaisseur du pli cutané du triceps (mm)
        - **Insulin** : Concentration en insuline (µU/mL)
        - **BMI** : Indice de masse corporelle
        - **DiabetesPedigreeFunction** : Antécédents familiaux (fonction d’hérédité)
        - **Age** : Âge du patient
        - **Outcome** : 1 = diabétique, 0 = non diabétique
        """)

    # Données pour l'entraînement
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrée utilisateur
    st.subheader("📝 Entrez vos informations médicales :")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pregnancies = st.number_input("Grossesses", 0, 20, step=1)
        insulin = st.number_input("Insuline (µU/mL)", 10, 1000, step=1)
    with col2:
        glucose = st.number_input("Glucose (mg/dL)", 50, 200, step=1)
        bmi = st.number_input("IMC (BMI)", 10.0, 50.0, step=0.1)
    with col3:
        blood_pressure = st.number_input("Pression Artérielle (mm Hg)", 40, 150, step=1)
        diabetes_pedigree = st.number_input("Hérédité", 0.0, 2.5, step=0.01)
    with col4:
        skin_thickness = st.number_input("Épaisseur de peau (mm)", 10, 80, step=1)
        age = st.number_input("Âge", 20, 100, step=1)

    # Prédiction
    if st.button("🚀 Lancer la prédiction"):
        model = RandomForestClassifier(
            n_estimators=n_arbre,
            max_depth=profondeur,
            bootstrap=boots,
            random_state=42
        )
        model.fit(X_train, y_train)

        user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                               insulin, bmi, diabetes_pedigree, age]])
        prediction = model.predict(user_data)
        prediction_proba = model.predict_proba(user_data)[:, 1]

        if prediction == 1:
            st.error("🔴 Vous êtes probablement diabétique.")
        else:
            st.success("🟢 Vous n'êtes probablement pas diabétique.")

        st.metric(label="Probabilité estimée de diabète", value=f"{prediction_proba[0]*100:.2f} %")

if __name__ == "__main__":
    main()
