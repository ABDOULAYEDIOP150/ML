import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score

def main():
    # Titre de l'application
    st.title("Application de Machine Learning pour la Prédiction du Diabète")
    st.subheader("Auteur: Diop Abdoulaye")

    # Fonction d'affichage de la matrice de confusion
    def plot_confusion_matrix(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel('Prédit')
        ax.set_ylabel('Réel')
        ax.set_title('Matrice de confusion')
        st.write("Matrice de confusion générée.")
        st.pyplot(fig)

    # Fonction d'affichage de la courbe ROC
    def plot_roc_curve(y_true, y_prob):
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f' % roc_auc)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlabel('Taux de faux positifs (FPR)')
        ax.set_ylabel('Taux de vrais positifs (TPR)')
        ax.set_title('Courbe ROC')
        ax.legend(loc="lower right")
        st.write("Courbe ROC générée.")
        st.pyplot(fig)

    # Fonction d'importation des données
    @st.cache_data
    def load_data():
        data = pd.read_csv(r"C:\Users\Abdoulaye Diop\Desktop\Projet_Data_Scientist_AtoZ\data\diabetes.csv")
        return data

    # Chargement des données
    df = load_data()

    # Division du dataset
    X = df.drop('Outcome', axis=1)  # Variables prédictives
    y = df['Outcome']  # Variable cible

    # Diviser la base de données en train et test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choix du modèle dans la sidebar
    classifier = st.sidebar.selectbox("Choisir un classificateur", ("Random Forest", "SVM", "Logistic Regression"))

    if classifier == "Random Forest":
        st.sidebar.subheader("Hyperparamètres du modèle Random Forest")
        n_arbre = st.sidebar.slider("Nombre d'arbres", 100, 1000, step=10)
        profondeur = st.sidebar.slider("Profondeur maximale de l'arbre", 1, 20, step=1)
        boots = st.sidebar.radio("Bootstrap lors de la création des arbres ?", ("True", "False"))
        boots = True if boots == "True" else False

    # Interface utilisateur pour entrer ses propres variables
    st.subheader("Entrez vos informations pour la prédiction")

    pregnancies = st.number_input("Nombre de grossesses", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Concentration en glucose (mg/dL)", min_value=50, max_value=200, step=1)
    blood_pressure = st.number_input("Pression artérielle (mm Hg)", min_value=40, max_value=150, step=1)
    skin_thickness = st.number_input("Épaisseur de la peau (mm)", min_value=10, max_value=80, step=1)
    insulin = st.number_input("Concentration en insuline (µU/mL)", min_value=10, max_value=1000, step=1)
    bmi = st.number_input("Indice de masse corporelle (BMI)", min_value=10.0, max_value=50.0, step=0.1)
    diabetes_pedigree = st.number_input("Diabetes pedigree function", min_value=0.0, max_value=2.5, step=0.01)
    age = st.number_input("Âge (années)", min_value=20, max_value=100, step=1)

    # Lorsque l'utilisateur clique sur le bouton pour faire une prédiction
    if st.button("Exécuter"):
        # Entraînement du modèle avec les hyperparamètres
        model = RandomForestClassifier(
            n_estimators=n_arbre,
            max_depth=profondeur,
            bootstrap=boots,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Prédiction avec les informations de l'utilisateur
        user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
        prediction = model.predict(user_data)
        prediction_proba = model.predict_proba(user_data)[:, 1]

        # Afficher le résultat de la prédiction
        if prediction == 1:
            st.write("### Vous êtes diagnostiqué avec le diabète.")
        else:
            st.write("### Vous n'êtes pas diagnostiqué avec le diabète.")

        st.write(f"Probabilité de diabète : {prediction_proba[0]:.2f}")

        # Optionnel: afficher la matrice de confusion et la courbe ROC
        if st.checkbox("Afficher la matrice de confusion"):
            st.write("Affichage de la matrice de confusion...")
            y_test_pred = model.predict(X_test)
            plot_confusion_matrix(y_test, y_test_pred)

        if st.checkbox("Afficher la courbe ROC"):
            st.write("Affichage de la courbe ROC...")
            y_test_prob = model.predict_proba(X_test)[:, 1]  # Probabilités pour la courbe ROC
            plot_roc_curve(y_test, y_test_prob)

if __name__ == "__main__":
    main()

