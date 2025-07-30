import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="📊 Tableau de Bord - Inventaire Entreprise",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé pour améliorer l'apparence
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .sidebar-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1rem 0;
    }
    
    .filter-section {
        background-color: #f1f3f4;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path):
    """Chargement et mise en cache des données"""
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {str(e)}")
        return None

def format_number(num):
    """Formatage des nombres pour l'affichage"""
    if pd.isna(num):
        return "N/A"
    if num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    else:
        return f"{num:.0f}"

def create_summary_metrics(df):
    """Création des métriques de résumé"""
    col1, col2, col3, col4 = st.columns(4)
    
    total_produits = len(df)
    total_montant = df['Montant (DA)'].sum() if 'Montant (DA)' in df.columns else 0
    total_quantite = df['Qte  (kg)'].sum() if 'Qte  (kg)' in df.columns else 0
    nb_lieux = df['LIEU'].nunique() if 'LIEU' in df.columns else 0
    
    with col1:
        st.metric(
            label="📦 Total Produits",
            value=format_number(total_produits),
            delta=f"{total_produits} articles"
        )
    
    with col2:
        st.metric(
            label="💰 Valeur Totale",
            value=f"{format_number(total_montant)} DA",
            delta="Montant inventaire"
        )
    
    with col3:
        st.metric(
            label="⚖️ Quantité Totale",
            value=f"{format_number(total_quantite)} kg",
            delta="Poids total"
        )
    
    with col4:
        st.metric(
            label="📍 Lieux de Stockage",
            value=nb_lieux,
            delta="Emplacements"
        )

def create_filters(df):
    """Création des filtres dans la sidebar"""
    st.sidebar.markdown('<div class="sidebar-header">🔍 Filtres de Données</div>', 
                       unsafe_allow_html=True)
    
    filters = {}
    
    # Filtre par lieu
    if 'LIEU' in df.columns:
        lieux = ['Tous'] + df['LIEU'].dropna().unique().tolist()
        filters['lieu'] = st.sidebar.selectbox("📍 Lieu de stockage", lieux)
    
    # Filtre par unité de travail
    if 'UNITE' in df.columns:
        unites = ['Toutes'] + df['UNITE'].dropna().unique().tolist()
        filters['unite'] = st.sidebar.selectbox("🏭 Unité de travail", unites)
    
    # Filtre par nature
    if 'NATURE' in df.columns:
        natures = ['Toutes'] + df['NATURE'].dropna().unique().tolist()
        filters['nature'] = st.sidebar.selectbox("🔖 Nature", natures)
    
    # Filtre par plage de montant
    if 'Montant (DA)' in df.columns:
        montant_min = float(df['Montant (DA)'].min())
        montant_max = float(df['Montant (DA)'].max())
        filters['montant_range'] = st.sidebar.slider(
            "💸 Plage de montant (DA)",
            montant_min, montant_max,
            (montant_min, montant_max)
        )
    
    return filters

def apply_filters(df, filters):
    """Application des filtres aux données"""
    filtered_df = df.copy()
    
    if 'lieu' in filters and filters['lieu'] != 'Tous':
        filtered_df = filtered_df[filtered_df['LIEU'] == filters['lieu']]
    
    if 'unite' in filters and filters['unite'] != 'Toutes':
        filtered_df = filtered_df[filtered_df['UNITE'] == filters['unite']]
    
    if 'nature' in filters and filters['nature'] != 'Toutes':
        filtered_df = filtered_df[filtered_df['NATURE'] == filters['nature']]
    
    if 'montant_range' in filters:
        min_val, max_val = filters['montant_range']
        filtered_df = filtered_df[
            (filtered_df['Montant (DA)'] >= min_val) & 
            (filtered_df['Montant (DA)'] <= max_val)
        ]
    
    return filtered_df

def create_visualizations(df):
    """Création des visualisations"""
    
    # Section 1: Analyse par lieu
    if 'LIEU' in df.columns and 'Montant (DA)' in df.columns:
        st.subheader("📍 Analyse par Lieu de Stockage")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Graphique en barres
            lieu_analysis = df.groupby('LIEU').agg({
                'Montant (DA)': 'sum',
                'Code Produit': 'count'
            }).reset_index()
            
            fig_bar = px.bar(
                lieu_analysis, 
                x='LIEU', 
                y='Montant (DA)',
                title="Valeur d'inventaire par lieu",
                color='Montant (DA)',
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(
                xaxis_tickangle=-45,
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Graphique en secteurs
            fig_pie = px.pie(
                lieu_analysis,
                values='Code Produit',
                names='LIEU',
                title="Répartition des produits par lieu"
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Section 2: Top produits
    st.subheader("🏆 Top Produits par Valeur")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 produits par montant
        if 'Montant (DA)' in df.columns:
            top_products = df.nlargest(10, 'Montant (DA)')
            
            fig_top = px.bar(
                top_products,
                x='Montant (DA)',
                y='Libellé Produit',
                orientation='h',
                title="Top 10 - Produits les plus chers",
                color='Montant (DA)',
                color_continuous_scale='reds'
            )
            fig_top.update_layout(height=500)
            st.plotly_chart(fig_top, use_container_width=True)
    
    with col2:
        # Analyse par famille de produit
        if 'FAMILLE DE PRODUIT' in df.columns:
            famille_analysis = df.groupby('FAMILLE DE PRODUIT').agg({
                'Montant (DA)': 'sum',
                'Qte  (kg)': 'sum'
            }).reset_index().sort_values('Montant (DA)', ascending=True)
            
            fig_famille = px.bar(
                famille_analysis,
                x='Montant (DA)',
                y='FAMILLE DE PRODUIT',
                orientation='h',
                title="Valeur par famille de produit",
                color='Qte  (kg)',
                color_continuous_scale='blues'
            )
            fig_famille.update_layout(height=500)
            st.plotly_chart(fig_famille, use_container_width=True)

def create_data_table(df):
    """Création du tableau de données interactif"""
    st.subheader("📋 Tableau de Données Détaillé")
    
    # Options d'affichage
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_rows = st.selectbox("Nombre de lignes à afficher", [10, 25, 50, 100, len(df)])
    
    with col2:
        columns_to_show = st.multiselect(
            "Colonnes à afficher",
            df.columns.tolist(),
            default=df.columns.tolist()[:8]  # Afficher les 8 premières colonnes par défaut
        )
    
    with col3:
        sort_column = st.selectbox("Trier par", df.columns.tolist())
    
    # Affichage du tableau filtré
    if columns_to_show:
        display_df = df[columns_to_show].sort_values(sort_column, ascending=False).head(show_rows)
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Bouton de téléchargement
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Télécharger les données filtrées (CSV)",
            data=csv,
            file_name=f'inventaire_filtre_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv'
        )

def create_pivot_analysis(df):
    """Création d'analyses de tableau croisé dynamique avec heatmap"""
    st.subheader("🔄 Tableau Croisé Dynamique & Heatmap")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sélection des paramètres du tableau croisé
        st.markdown("### ⚙️ Configuration du Tableau Croisé")
        
        # Colonnes disponibles pour les analyses
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Sélecteurs pour le tableau croisé
        index_col = st.selectbox(
            "📋 Lignes (Index)", 
            categorical_cols,
            help="Choisissez la colonne pour les lignes du tableau"
        )
        
        columns_col = st.selectbox(
            "📊 Colonnes", 
            categorical_cols,
            help="Choisissez la colonne pour les colonnes du tableau"
        )
        
        values_col = st.selectbox(
            "💰 Valeurs", 
            numeric_cols,
            help="Choisissez la colonne pour les valeurs à agréger"
        )
        
        aggfunc = st.selectbox(
            "📈 Fonction d'agrégation",
            ["sum", "mean", "count", "median", "std"],
            help="Choisissez comment agréger les données"
        )
        
        show_heatmap = st.checkbox("🌡️ Afficher la Heatmap", value=True)
        show_totals = st.checkbox("🔢 Afficher les totaux", value=True)
    
    with col2:
        # Options de personnalisation de la heatmap
        st.markdown("### 🎨 Options d'Affichage")
        
        color_scale = st.selectbox(
            "🎨 Palette de couleurs",
            ["Viridis", "RdBu", "Blues", "Reds", "YlOrRd", "Plasma", "Cividis"],
            help="Choisissez la palette de couleurs pour la heatmap"
        )
        
        show_values = st.checkbox("📝 Afficher les valeurs", value=True)
        format_numbers = st.checkbox("🔢 Formater les nombres", value=True)
        
        # Format d'affichage des nombres
        if format_numbers:
            decimal_places = st.slider("Décimales", 0, 3, 1)
        else:
            decimal_places = 2
    
    # Création du tableau croisé dynamique
    try:
        if index_col and columns_col and values_col:
            # Création du pivot table
            pivot_table = pd.pivot_table(
                df, 
                index=index_col, 
                columns=columns_col, 
                values=values_col, 
                aggfunc=aggfunc,
                fill_value=0
            )
            
            # Ajout des totaux si demandé
            if show_totals:
                # Calcul des totaux par ligne
                pivot_table['Total'] = pivot_table.sum(axis=1)
                
                # Calcul des totaux par colonne
                totals_row = pivot_table.sum(axis=0)
                totals_row.name = 'Total'
                pivot_table = pd.concat([pivot_table, totals_row.to_frame().T])
            
            # Affichage du tableau
            st.markdown("### 📊 Résultats du Tableau Croisé")
            
            # Formatage des nombres pour l'affichage
            if format_numbers:
                formatted_pivot = pivot_table.round(decimal_places)
                # Format avec séparateur de milliers
                display_pivot = formatted_pivot.applymap(
                    lambda x: f"{x:,.{decimal_places}f}".replace(',', ' ') if pd.notnull(x) else ""
                )
            else:
                display_pivot = pivot_table
            
            # Affichage du tableau avec style
            st.dataframe(
                display_pivot.style.highlight_max(axis=1, color='lightgreen')
                                  .highlight_min(axis=1, color='lightcoral'),
                use_container_width=True,
                height=400
            )
            
            # Bouton de téléchargement du tableau
            csv_pivot = pivot_table.to_csv()
            st.download_button(
                label="📥 Télécharger le tableau croisé (CSV)",
                data=csv_pivot,
                file_name=f'tableau_croise_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
            
            # Affichage de la heatmap si demandé
            if show_heatmap:
                st.markdown("### 🌡️ Heatmap Interactive")
                
                # Préparation des données pour la heatmap (sans les totaux pour une meilleure lisibilité)
                heatmap_data = pivot_table.copy()
                if show_totals:
                    # Enlever la ligne et colonne des totaux pour la heatmap
                    if 'Total' in heatmap_data.columns:
                        heatmap_data = heatmap_data.drop('Total', axis=1)
                    if 'Total' in heatmap_data.index:
                        heatmap_data = heatmap_data.drop('Total', axis=0)
                
                # Création de la heatmap avec Plotly
                fig_heatmap = px.imshow(
                    heatmap_data,
                    text_auto=show_values,
                    aspect="auto",
                    title=f"Heatmap : {index_col} vs {columns_col} ({values_col})",
                    color_continuous_scale=color_scale.lower(),
                    labels=dict(x=columns_col, y=index_col, color=f"{aggfunc}({values_col})")
                )
                
                # Personnalisation de la heatmap
                fig_heatmap.update_layout(
                    height=max(400, len(heatmap_data) * 30),  # Hauteur dynamique
                    xaxis_title=columns_col,
                    yaxis_title=index_col,
                    font_size=10
                )
                
                # Rotation des labels si nécessaire
                if len(heatmap_data.columns) > 5:
                    fig_heatmap.update_xaxes(tickangle=45)
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Statistiques sur la heatmap
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📊 Valeur Max", f"{heatmap_data.max().max():.{decimal_places}f}")
                with col2:
                    st.metric("📉 Valeur Min", f"{heatmap_data.min().min():.{decimal_places}f}")
                with col3:
                    st.metric("📈 Moyenne", f"{heatmap_data.mean().mean():.{decimal_places}f}")
                with col4:
                    st.metric("🎯 Médiane", f"{heatmap_data.median().median():.{decimal_places}f}")
            
    except Exception as e:
        st.error(f"❌ Erreur lors de la création du tableau croisé : {str(e)}")
        st.info("💡 Vérifiez que vous avez sélectionné des colonnes appropriées pour l'analyse.")

def create_advanced_analysis(df):
    """Analyses avancées"""
    st.subheader("📊 Analyses Avancées")
    
    tab1, tab2, tab3, tab4 = st.tabs(["Tableau Croisé", "Corrélations", "Distribution", "Analyse Personnalisée"])
    
    with tab1:
        # Nouveau tableau croisé dynamique avec heatmap
        create_pivot_analysis(df)
    
    with tab2:
        # Matrice de corrélation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Matrice de Corrélation",
                color_continuous_scale='RdBu'
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab3:
        # Distribution des variables numériques
        if 'Montant (DA)' in df.columns:
            fig_hist = px.histogram(
                df,
                x='Montant (DA)',
                nbins=30,
                title="Distribution des montants",
                marginal="box"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab4:
        # Analyse personnalisée
        st.write("Créez vos propres visualisations :")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox("Axe X", df.columns.tolist(), key="custom_x")
            y_axis = st.selectbox("Axe Y", df.columns.tolist(), key="custom_y")
        
        with col2:
            chart_type = st.selectbox("Type de graphique", 
                                    ["Scatter", "Bar", "Line", "Box"])
            color_by = st.selectbox("Couleur par", ["Aucune"] + df.columns.tolist())
        
        if st.button("Générer le graphique"):
            color_col = None if color_by == "Aucune" else color_by
            
            if chart_type == "Scatter":
                fig = px.scatter(df, x=x_axis, y=y_axis, color=color_col)
            elif chart_type == "Bar":
                fig = px.bar(df, x=x_axis, y=y_axis, color=color_col)
            elif chart_type == "Line":
                fig = px.line(df, x=x_axis, y=y_axis, color=color_col)
            else:  # Box
                fig = px.box(df, x=x_axis, y=y_axis, color=color_col)
            
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Fonction principale"""
    
    # En-tête
    st.markdown('<div class="main-header">📊 Tableau de Bord - Inventaire Entreprise</div>', 
                unsafe_allow_html=True)
    
    # Upload de fichier
    st.sidebar.markdown('<div class="sidebar-header">📁 Chargement des Données</div>', 
                       unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader(
        "Choisir un fichier Excel",
        type=['xlsx', 'xls'],
        help="Téléchargez votre fichier d'inventaire au format Excel"
    )
    
    # Chargement des données
    df = None
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
    else:
        # Option pour utiliser un fichier par défaut (pour les tests)
        if st.sidebar.checkbox("Utiliser le fichier par défaut"):
            try:
                df = load_data('C:/Users/SOL/Downloads/df_fini_modifié (1).xlsx')
            except:
                st.error("Fichier par défaut non trouvé. Veuillez télécharger un fichier.")
    
    if df is not None and not df.empty:
        # Affichage des informations sur le dataset
        st.info(f"📈 Dataset chargé : {len(df)} lignes, {len(df.columns)} colonnes")
        
        # Création des filtres
        filters = create_filters(df)
        
        # Application des filtres
        filtered_df = apply_filters(df, filters)
        
        # Vérification si des données restent après filtrage
        if len(filtered_df) == 0:
            st.warning("⚠️ Aucune donnée ne correspond aux filtres sélectionnés.")
            return
        
        # Métriques de résumé
        create_summary_metrics(filtered_df)
        
        # Visualisations
        create_visualizations(filtered_df)
        
        # Tableau de données
        create_data_table(filtered_df)
        
        # Analyses avancées
        create_advanced_analysis(filtered_df)
        
    else:
        st.info("👆 Veuillez télécharger un fichier Excel pour commencer l'analyse.")
        
        # Affichage des colonnes attendues
        st.markdown("### 📋 Format de fichier attendu")
        st.markdown("""
        Votre fichier Excel doit contenir les colonnes suivantes :
        
        - **Code Produit** : Identifiant unique du produit
        - **Libellé Produit** : Description du produit
        - **Montant (DA)** : Valeur en dinars algériens
        - **Qte (kg)** : Quantité en kilogrammes
        - **LIEU** : Lieu de stockage
        - **UNITE** : Unité de travail
        - **NATURE** : Type de produit
        - *Et autres colonnes selon vos besoins...*
        """)

if __name__ == "__main__":
    main()
