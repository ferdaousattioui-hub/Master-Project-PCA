#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA

# =====================================================================
# 1. CONFIGURATION ESTHÉTIQUE SÉCURISÉE (Zéro bug de style)
# =====================================================================
try:
    plt.style.use('seaborn-whitegrid')
except:
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')

# Customisation pour un rendu premium / moderne
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['grid.color'] = '#e9ecef'
font_title = {'color': '#1a1a1a', 'weight': 'bold', 'size': 14}

# =====================================================================
# 2. CHARGEMENT ET PRÉPARATION DU DATASET MNIST
# =====================================================================
print("🔄 Chargement des données MNIST (Extraction de 5000 images)...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_raw = mnist.data[:5000] / 255.0  # Normalisation des pixels entre 0 et 1

# =====================================================================
# 3. INJECTION D'UN BRUIT BLANC GAUSSIEN INTENSE
# =====================================================================
np.random.seed(2026)
# On passe de 0.55 à 0.25 pour redonner l'avantage au signal du chiffre
bruit_optimal = np.random.normal(loc=0.0, scale=0.25, size=X_raw.shape) 
X_bruite = np.clip(X_raw + bruit_optimal, 0., 1.)

# =====================================================================
# 4. TRANSFORMATION PCA : LE JUSTE MILIEU
# =====================================================================
# Option A : On cible 75% de variance pour forcer l'exclusion du bruit
pca = PCA(n_components=0.75) 

# Option B (Alternative encore plus radicale) : On force un nombre fixe d'axes
# pca = PCA(n_components=40) 

X_reduit = pca.fit_transform(X_bruite)
X_debruite = np.clip(pca.inverse_transform(X_reduit), 0., 1.)

print("\n✅ Analyse et traitement PCA terminés avec succès !")
print(f"   🔹 Espace d'origine  : {X_raw.shape[1]} dimensions (pixels).")
print(f"   🔹 Espace de la PCA  : {pca.n_components_} composantes principales retenues.")
print(f"   🔹 Taux de compression : {round((1 - (pca.n_components_ / 784)) * 100, 2)}% de dimensions économisées.\n")

# =====================================================================
# 5. VISUALISATION 1 : LE COMPARATIF TRIPLE LIGNE (Effet Waouh)
# =====================================================================
n_digits = 6  # Nombre de chiffres à afficher en colonnes
fig, axes = plt.subplots(3, n_digits, figsize=(14, 8.5))

# Palette de couleurs 'magma' pour un effet thermique très pro
cmap_choice = 'magma' 

for i in range(n_digits):
    # Ligne 1 : Image Originale (Le Signal Pur)
    axes[0, i].imshow(X_raw[i].reshape(28, 28), cmap=cmap_choice)
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title("1. SIGNAL PUR\n(MNIST Original)", fontdict=font_title, loc='left')

    # Ligne 2 : Image Noyée sous le Bruit (Friture TV complète)
    axes[1, i].imshow(X_bruite[i].reshape(28, 28), cmap=cmap_choice)
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title("2. BRUIT BLANC INTENSE\n(Signal totalement noyé)", fontdict=font_title, loc='left')

    # Ligne 3 : Restauration Géométrique par la PCA
    axes[2, i].imshow(X_debruite[i].reshape(28, 28), cmap=cmap_choice)
    axes[2, i].axis('off')
    if i == 0:
        axes[2, i].set_title("3. FILTRAGE VIA PCA\n(Restauration des structures)", fontdict=font_title, loc='left')

plt.tight_layout()
plt.subplots_adjust(hspace=0.45)
print("📊 Génération de la matrice comparative graphique...")
plt.show()

# =====================================================================
# 6. VISUALISATION 2 : COURBE DE SCREENING (L'explication pour le Jury)
# =====================================================================
plt.figure(figsize=(10, 5.5))
variance_cumulee = np.cumsum(pca.explained_variance_ratio_)

# Coloration esthétique de la zone d'information utile
plt.fill_between(range(len(variance_cumulee)), variance_cumulee, color="#8e44ad", alpha=0.15)

# Tracé de la courbe et des lignes de repère
plt.plot(variance_cumulee, color="#8e44ad", linewidth=3, label="Variance expliquée cumulée")
plt.axhline(y=0.90, color='#e74c3c', linestyle='--', linewidth=2, label='Seuil d\'Information Ciblé (90%)')
plt.axvline(x=pca.n_components_, color='#2ecc71', linestyle=':', linewidth=2, label=f'{pca.n_components_} Axes Sélectionnés')

# Habillage du graphique
plt.title("Pourquoi la PCA élimine-t-elle le bruit ? (Analyse Spectrale)", fontsize=14, fontweight='bold', pad=15)
plt.xlabel("Nombre de Composantes Principales (Nouvel Espace Vectoriel)", fontsize=11)
plt.ylabel("Ratio de Variance Capturée", fontsize=11)
plt.xlim(0, 150)  # Zoom sur la zone du "coude" pour une meilleure lisibilité
plt.ylim(0, 1.05)
plt.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()

print("📊 Génération de la courbe de variance...")
plt.show()


# In[ ]:




