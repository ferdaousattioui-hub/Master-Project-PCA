import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Projet ML - PCA Dénoiser", layout="wide")

# --- BARRE LATÉRALE DE NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Sélectionnez ce que vous voulez voir :", ["1. Présentation PPT (PDF)", "2. Cas Pratique PCA (MNIST)"])

# --- CONFIGURATION ESTHÉTIQUE DES GRAPHES (Premium Style) ---
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['grid.color'] = '#e9ecef'
font_title = {'color': '#1a1a1a', 'weight': 'bold', 'size': 16}

# --- SECTION 1 : AFFICHAGE DU PPT ---
if page == "1. Présentation PPT (PDF)":
    st.title("📂 Présentation du Projet - PCA")
    st.write("Voici les diapositives théoriques de mon projet sur l'Analyse en Composantes Principales.")
    
    # Lien de ton Google Drive configuré en Public
    file_id = "1yYGQ78AofpLGR8kiaYVeSa4gTHx2zQB4"
    lien_drive_embed = f"https://drive.google.com/file/d/{file_id}/preview"
    
    st.components.v1.html(
        f'<iframe src="{lien_drive_embed}" style="width:100%; height:750px;" frameborder="0" allowfullscreen></iframe>',
        height=750
    )

# --- SECTION 2 : CAS PRATIQUE INTERACTIF  ---
elif page == "2. Cas Pratique PCA (MNIST)":
    st.title("💻 Application Interactive - Dénoyage d'images par PCA")
    st.write("Cette application démontre la puissance géométrique de la PCA pour filtrer le bruit blanc intense d'un signal.")

    # --- BARRE LATÉRALE DES PARAMÈTRES (INTERACTIF POUR LE JURY) ---
    st.sidebar.subheader("🎛️ Paramètres en Direct")
    variance_cible = st.sidebar.slider("Variance expliquée ciblée (%)", min_value=50, max_value=95, value=75, step=5) / 100.0
    niveau_bruit = st.sidebar.slider("Intensité du bruit gaussien", min_value=0.1, max_value=0.6, value=0.25, step=0.05)

    # Chargement ULTRA-RAPIDE pour éviter le crash Streamlit Cloud
    digits = load_digits()
    X_raw = digits.data / 16.0  

    # Injection du bruit blanc gaussien intense 
    np.random.seed(2026)
    bruit = np.random.normal(loc=0.0, scale=niveau_bruit, size=X_raw.shape) 
    X_bruite = np.clip(X_raw + bruit, 0., 1.)

    # Transformation PCA
    pca = PCA(n_components=variance_cible) 
    X_reduit = pca.fit_transform(X_bruite)
    X_debruite = np.clip(pca.inverse_transform(X_reduit), 0., 1.)

    # --- PANNEAU DE STATISTIQUES MODERNES ---
    col1, col2, col3 = st.columns(3)
    col1.metric(label="📊 Espace d'origine", value=f"{X_raw.shape[1]} Pixels (Dimensions)")
    col2.metric(label="⚙️ Composantes retenues", value=f"{pca.n_components_} Axes Vectoriels")
    taux_compression = round((1 - (pca.n_components_ / X_raw.shape[1])) * 100, 2)
    col3.metric(label="📉 Dimensions économisées", value=f"{taux_compression} %")

    st.markdown("---")

    # --- VISUALISATION 1 : LE COMPARATIF TRIPLE LIGNE (EFFET WAOUH!) ---
    st.subheader("🔥 Matrice de Restauration : Signal Pur vs Bruité vs Filtré")
    
    n_digits = 6 
    # T-7kem f figsize bach tban kbira o mferg3a f l-interface
    fig1, axes1 = plt.subplots(3, n_digits, figsize=(15, 9.5))
    cmap_choice = 'magma' # Dik l-palette dyal magma lli khtariyti premium bzaff!

    for i in range(n_digits):
        # Ligne 1 : Image Originale (Le Signal Pur)
        axes1[0, i].imshow(X_raw[i].reshape(8, 8), cmap=cmap_choice)
        axes1[0, i].axis('off')
        if i == 0:
            axes1[0, i].set_title("1. SIGNAL PUR\n(MNIST Original)", fontdict=font_title, loc='left')

        # Ligne 2 : Image Noyée sous le Bruit
        axes1[1, i].imshow(X_bruite[i].reshape(8, 8), cmap=cmap_choice)
        axes1[1, i].axis('off')
        if i == 0:
            axes1[1, i].set_title("2. BRUIT BLANC INTENSE\n(Signal noyé)", fontdict=font_title, loc='left')

        # Ligne 3 : Restauration Géométrique par la PCA
        axes1[2, i].imshow(X_debruite[i].reshape(8, 8), cmap=cmap_choice)
        axes1[2, i].axis('off')
        if i == 0:
            axes1[2, i].set_title("3. FILTRAGE VIA PCA\n(Structures restaurées)", fontdict=font_title, loc='left')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)
    st.pyplot(fig1) # Afichage direct 

    st.markdown("---")

    # --- VISUALISATION 2 : COURBE DE SCREENING 
    st.subheader("📈 Courbe de Screening & Analyse Spectrale du Bruit")
    
    fig2 = plt.figure(figsize=(11, 5.5))
    variance_cumulee = np.cumsum(pca.explained_variance_ratio_)

    plt.fill_between(range(len(variance_cumulee)), variance_cumulee, color="#8e44ad", alpha=0.15)
    plt.plot(variance_cumulee, color="#8e44ad", linewidth=3, label="Variance expliquée cumulée")
    plt.axhline(y=variance_cible, color='#e74c3c', linestyle='--', linewidth=2, label=f'Seuil cible ({int(variance_cible*100)}%)')
    plt.axvline(x=pca.n_components_, color='#2ecc71', linestyle=':', linewidth=2, label=f'{pca.n_components_} Axes Sélectionnés')

    plt.title("Pourquoi la PCA élimine-t-elle le bruit ?", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Nombre de Composantes Principales (Nouvel Espace Vectoriel)", fontsize=11)
    plt.ylabel("Ratio de Variance Capturée", fontsize=11)
    plt.xlim(0, X_raw.shape[1]) 
    plt.ylim(0, 1.05)
    plt.legend(loc="lower right", frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    
    st.pyplot(fig2)
    st.success(f"✅ Analyse terminée ! En ciblant {int(variance_cible*100)}% de l'information, la PCA isole les structures géométriques essentielles et rejette le bruit mathématique résiduel.")
