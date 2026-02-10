# ==============================================================
# 1. Chargement et inspection des données
# ==============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configuration graphique
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10,6)

print(" Bibliothèques importées avec succès")


# CHARGEMENT DU DATASET
housing = fetch_california_housing(as_frame=True)
df = housing.frame

print("Dimensions du dataset :", df.shape)
print(f"Dimensions du dataset : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"\nVariable cible : MedHouseVal (valeur médiane du logement en centaines de milliers de $)")
print(f"\nDescription des variables :")
print("-" * 60)
descriptions = {
'MedInc': 'Revenu médian du quartier',
'HouseAge': 'Âge médian des logements',
'AveRooms': 'Nombre moyen de pièces par logement',
'AveBedrms': 'Nombre moyen de chambres par logement',
'Population': 'Population du quartier',
'AveOccup': 'Nombre moyen d\'occupants par logement',
'Latitude': 'Latitude géographique',
'Longitude': 'Longitude géographique',
'MedHouseVal': 'Prix médian du logement (cible)'
}
for col, desc in descriptions.items():
    print(f" {col:15s} → {desc}")

# Afficherage les premières lignes avec .head()
df.head()
# Types de variables + valeurs manquantes
df.info()

# Statistiques descriptives
df.describe()

# Vérification des valeurs manquantes
df.isnull().sum()

# ==============================================================
# 2.  NETTOYAGE DES DONNEES
# ==============================================================

# Vérification des valeurs NULL
df.isnull().sum()
# SUPPRESSION DES DOUBLONS
print("Doublons :", df.duplicated().sum())
df = df.drop_duplicates()
# Boxplot pour visualiser les valeurs aberrantes
plt.figure()
sns.boxplot(data=df[['AveRooms','AveBedrms','AveOccup']])
plt.title("Détection des outliers")
plt.show()

# Suppression des valeurs abberantes
df = df[df['AveRooms'] < 50]
df = df[df['AveBedrms'] < 20]
df = df[df['AveOccup'] < 20]

print("Nouvelle dimension :", df.shape)

# ==============================================================
# 3. EDA
# ==============================================================
# Histogramme + courbe KDE
sns.histplot(df['MedHouseVal'], kde=True)
plt.title("Distribution du Prix Médian des Logements")
plt.show()
# Boxplot
sns.boxplot(x=df['MedHouseVal'])
plt.title("Boxplot du Prix Médian")
plt.show()
# Matrice de correlation
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Matrice de Corrélation")
plt.show()
# Relation revenu / prix
sns.scatterplot(x='MedInc', y='MedHouseVal', data=df)
plt.title("Relation entre Revenu Médian et Prix")
plt.show()
# Carte géographique des prix
plt.scatter(df['Longitude'], df['Latitude'],c=df['MedHouseVal'], cmap='viridis')
plt.colorbar(label='Prix Médian')
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Répartition Géographique des Prix")
plt.show()

# ==============================================================
# 4. INGENIERIE DES VARIABLES
# ==============================================================

# Creation des nouvelles variables
df['PiecesParChambre'] = df['AveRooms'] / df['AveBedrms']
df['PopParLogement'] = df['Population'] / df['AveOccup']
df.head()
# Séparation des variables explicatives et cible
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']
print("Nombre de variables explicatives :", X.shape[1])

# ==============================================================
# 5. Séparation entraînement / test
# ==============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Taille train :", X_train.shape)
print("Taille test :", X_test.shape)

# ==============================================================
# 6. CONSTRUCTION DU MODÈLE
# ==============================================================

model = LinearRegression()
model.fit(X_train, y_train)
print("Intercept (β0) :", model.intercept_)
# Affichage des coefficients
coef_df = pd.DataFrame({
    "Variable": X.columns,
    "Coefficient": model.coef_
})
coef_df.sort_values(by="Coefficient", ascending=False)

# ==============================================================
#7 . PRÉDICTIONS et EVALUATIONS
# ==============================================================

y_pred = model.predict(X_test)
# Metrique d'evaluation
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("MSE  :", mse)
print("RMSE :", rmse)
print("MAE  :", mae)
print("R²   :", r2)

# ==============================================================
# 8. VISUALISATION DES RESULTATS
# ==============================================================

# Graphique 1 : Valeurs réelles vs Prédites
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')
plt.xlabel("Valeurs Réelles")
plt.ylabel("Valeurs Prédites")
plt.title("Réel vs Prédit")
plt.show()
# Graphique 2 : Analyse des résidus (résidu = réel - prédit)
residus = y_test - y_pred
sns.histplot(residus, kde=True)
plt.title("Distribution des Résidus")
plt.show()
plt.scatter(y_pred, residus)
plt.axhline(0, color='red')
plt.xlabel("Prédictions")
plt.ylabel("Résidus")
plt.title("Résidus vs Prédictions")
plt.show()
# Graphique 3 : Barplot des coefficients du modèle
sns.barplot(x="Coefficient", y="Variable", data=coef_df)
plt.title("Importance des Variables")
plt.show()

# ==============================================================
# 9. EXPORT DES RÉSULTATS
# ==============================================================

results = X_test.copy()
results['Prix_Reel'] = y_test
results['Prix_Predit'] = y_pred
results['Residus'] = results['Prix_Reel'] - results['Prix_Predit']
results['Erreur_%'] = abs(results['Residus'] / results['Prix_Reel']) * 100
results.to_csv("resultats_regression.csv", index=False)
print("Fichier exporté avec succès")
# Téléchargement du fichier
from google.colab import files
files.download("resultats_regression.csv")


# ==============================================================
# 10. ANALYSE et RECOMMANDATION
# ==============================================================

# ---- Analyse des résultats -----

# Le modèle de régression linéaire montre que la variable MedInc (revenu médian) est celle qui influence le plus le prix des logements.
# Les zones avec un revenu plus élevé ont généralement des prix plus élevés.

# Le coefficient de détermination R² = ... indique que le modèle explique environ (R² × 100)% de la variance des prix.

# Les résidus sont globalement centrés autour de 0, ce qui signifie que le modèle est globalement adapté.

# --- Recommandations  -----

# Ajouter des modèles plus avancés (Ridge, Lasso, Random Forest)

# Standardiser les variables

# Ajouter des interactions entre variables

# Tester une validation croisée
