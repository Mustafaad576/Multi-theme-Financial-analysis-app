import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, silhouette_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import plotly.express as px
import plotly.graph_objects as go
# import matplotlib.pyplot as plt # Not directly used for plots
import joblib # For saving/loading models
import io # For downloading files
import warnings

# Suppress specific warnings if needed (optional)
# warnings.filterwarnings("ignore", category=FutureWarning)

# --- Page Configuration (Theme & Layout) ---
st.set_page_config(
    page_title="Lannister Financial Analysis",
    page_icon="🦁", # Lannister Lion
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Lannister Theme ---
LANNISTER_CRIMSON = "#8B0000" # Dark Red
LANNISTER_GOLD = "#FFD700"    # Gold
LANNISTER_DARK = "#1A1A1A"    # Very Dark Gray/Almost Black Background
LANNISTER_TEXT = "#EAEAEA"    # Light Gray/Off-White Text
LANNISTER_TEXT_HEADER = "#FFFFFF" # Brighter White for Headers
LANNISTER_SIDEBAR = "#2B2B2B" # Darker Gray Sidebar
LANNISTER_BORDER = "#444444" # Dark Border

# --- Font ---
GOOGLE_FONT = "Cinzel" # Google Font for Lannister theme
FONT_URL = f"https://fonts.googleapis.com/css2?family={GOOGLE_FONT.replace(' ', '+')}:wght@400;700&display=swap"

# --- Custom CSS for Lannister Theming ---
st.markdown(f"""
<link href="{FONT_URL}" rel="stylesheet">
<style>
    /* Apply Font Globally */
    html, body, [class*="st-"], button, input, select, textarea {{
        font-family: '{GOOGLE_FONT}', serif !important;
    }}

    /* Main background and text */
    .stApp {{
        background-color: {LANNISTER_DARK};
        color: {LANNISTER_TEXT};
    }}

    /* Sidebar */
    [data-testid="stSidebar"] > div:first-child {{
        background-color: {LANNISTER_SIDEBAR};
        border-right: 1px solid {LANNISTER_BORDER};
    }}
    /* Sidebar Headers & Text */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stText,
    [data-testid="stSidebar"] .stRadio label span {{ /* Target radio labels */
        color: {LANNISTER_TEXT_HEADER} !important; /* Brighter text for sidebar */
        font-weight: 700; /* Bolder */
    }}
    /* Sidebar input widgets */
    [data-testid="stSidebar"] .stTextInput input {{
        background-color: #383838 !important;
        color: {LANNISTER_TEXT} !important;
        border: 1px solid {LANNISTER_BORDER} !important;
    }}
    /* --- CSS Fix for Sidebar Selectbox Text Visibility --- */
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
        background-color: #383838 !important;
        color: {LANNISTER_TEXT} !important;
        border: 1px solid {LANNISTER_BORDER} !important;
        padding: 0.5em 0.6em !important; /* Adjusted padding */
        height: auto !important; /* Let height adjust to content */
        min-height: 40px !important; /* Ensure a minimum clickable height */
        line-height: 1.6 !important; /* Ensure line height allows text */
    }}
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div > div {{
         /* This targets the inner div holding the text */
         color: {LANNISTER_TEXT} !important;
         overflow: visible !important; /* Allow text to overflow if needed (shouldn't be necessary with auto height) */
         white-space: normal !important; /* Allow text wrapping if needed */
         height: auto !important;
    }}
    /* --- End of CSS Fix --- */

    [data-testid="stSidebar"] .stFileUploader label span {{
         color: {LANNISTER_TEXT} !important; /* Ensure file uploader text is visible */
    }}


    /* Buttons */
    .stButton>button {{
        background-color: {LANNISTER_CRIMSON};
        color: {LANNISTER_GOLD}; /* Gold text on Crimson button */
        border: 1px solid {LANNISTER_GOLD};
        border-radius: 5px;
        padding: 10px 22px;
        transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
        font-weight: 700; /* Bold */
        font-size: 1.05rem;
        text-transform: uppercase; /* Lannister style */
        letter-spacing: 0.5px;
    }}
    .stButton>button:hover {{
        background-color: {LANNISTER_GOLD}; /* Gold background on hover */
        color: {LANNISTER_CRIMSON}; /* Crimson text on hover */
        border-color: {LANNISTER_CRIMSON};
    }}
     .stButton>button:focus {{
         box-shadow: 0 0 0 3px rgba(255, 215, 0, 0.4); /* Gold focus ring */
         outline: none;
    }}

    /* Titles and Headers in Main Area */
    h1, h2, h3 {{
        color: {LANNISTER_GOLD}; /* Gold Headers */
        font-weight: 700;
        text-shadow: 1px 1px 2px #000000; /* Subtle shadow */
    }}
    h1 {{
        border-bottom: 2px solid {LANNISTER_CRIMSON};
        padding-bottom: 0.6rem;
        text-align: center; /* Center main title */
    }}
     h3 {{ /* Step headers */
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-top: 1px solid {LANNISTER_BORDER};
        padding-top: 1rem;
        color: {LANNISTER_GOLD}; /* Ensure step headers are gold */
    }}


    /* Dataframe styling */
     .stDataFrame {{
        border: 1px solid {LANNISTER_BORDER};
        border-radius: 0px; /* Sharp corners */
        background-color: #252525; /* Dark background for tables */
    }}
     /* Dataframe header */
    .stDataFrame thead th {{
        background-color: {LANNISTER_CRIMSON};
        color: {LANNISTER_GOLD};
        font-weight: 700;
        text-transform: uppercase;
    }}
    /* Dataframe cells */
    .stDataFrame tbody td {{
        color: {LANNISTER_TEXT};
    }}

    /* Style metric labels/values */
    [data-testid="stMetricLabel"] {{
        color: #AAAAAA; /* Lighter gray for metric labels */
        font-size: 0.95rem;
        text-transform: uppercase;
    }}
    [data-testid="stMetricValue"] {{
        color: {LANNISTER_GOLD}; /* Gold metric value */
        font-size: 1.3rem;
        font-weight: 700;
    }}

    /* Input widgets styling in main area */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div,
    .stMultiSelect>div>div {{ /* Target multiselect */
        border: 1px solid {LANNISTER_BORDER} !important;
        border-radius: 0px; /* Sharp corners */
        background-color: #303030 !important;
        color: {LANNISTER_TEXT} !important;
    }}
     /* Adjust padding/appearance for select/multiselect */
    .stSelectbox>div>div, .stMultiSelect>div>div {{
         padding-top: 0.4rem;
         padding-bottom: 0.4rem;
    }}
    /* Slider track/thumb */
    .stSlider [data-baseweb="slider"] > div:nth-child(2) > div {{ /* Track */
         background: {LANNISTER_CRIMSON};
    }}
     .stSlider [data-baseweb="slider"] > div:nth-child(3) {{ /* Thumb */
         background: {LANNISTER_GOLD};
         border: 2px solid {LANNISTER_CRIMSON};
    }}

    /* Style info/success/warning/error boxes */
    .stAlert {{
         border-radius: 0px; /* Sharp corners */
         border: 1px solid {LANNISTER_GOLD};
         background-color: rgba(0, 0, 0, 0.3); /* Dark transparent background */
    }}
    [data-testid="stAlert"] p {{
         color: {LANNISTER_TEXT};
    }}
    /* Specific alert icons/accents (using border color for theme) */
    [data-testid="stAlert"][kind="info"] {{ border-left: 5px solid #4a90e2; }} /* Blue accent */
    [data-testid="stAlert"][kind="success"] {{ border-left: 5px solid #50e3c2; }} /* Green accent */
    [data-testid="stAlert"][kind="warning"] {{ border-left: 5px solid {LANNISTER_GOLD}; }} /* Gold accent */
    [data-testid="stAlert"][kind="error"] {{ border-left: 5px solid {LANNISTER_CRIMSON}; }} /* Crimson accent */

    /* Center welcome screen elements */
    .welcome-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        /* min-height removed to reduce vertical space */
        text-align: center;
        padding-top: 2rem; /* Add some padding at the top */
        padding-bottom: 2rem; /* Add padding at bottom */
    }}

</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---#

def download_file(data, filename, label, file_format='csv'):
    """Generates a download button for dataframes or models."""
    try:
        buffer = io.BytesIO()
        if file_format == 'csv' and isinstance(data, pd.DataFrame):
            if data.empty: st.warning(f"Cannot download {label}: Dataframe is empty."); return
            data.to_csv(buffer, index=False)
            mime = 'text/csv'
        elif file_format == 'joblib':
            if data is None: st.error(f"Cannot download {label}: Data is None."); return
            joblib.dump(data, buffer)
            mime = 'application/octet-stream'
        else: st.error(f"Unsupported download format for {label}: {file_format}"); return

        buffer.seek(0)
        st.download_button(
            label=f"Hear Me Roar! (Download {label})", # Themed label
            data=buffer,
            file_name=filename,
            mime=mime,
            key=f"download_{filename.replace('.', '_')}"
        )
    except Exception as e: st.error(f"Error preparing {label} for download: {e}")


def plot_feature_importance(model, feature_names):
    """Plots feature importance for linear models."""
    importance = None
    if hasattr(model, 'coef_'):
        if model.coef_.ndim == 1: importance = model.coef_
        elif model.coef_.ndim == 2: importance = np.abs(model.coef_).max(axis=0)
        else: st.warning("Could not determine feature importance structure."); return None
    elif hasattr(model, 'feature_importances_'): importance = model.feature_importances_
    else: st.info("Feature importance plotting not available for this model type."); return None

    if importance is None or len(feature_names) != len(importance):
        st.warning(f"Feature names length ({len(feature_names)}) != importance length ({len(importance) if importance is not None else 'N/A'}). Skipping plot."); return None

    importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    importance_df['abs_importance'] = np.abs(importance_df['importance'])
    importance_df = importance_df.sort_values('abs_importance', ascending=False).head(15)

    fig = px.bar(importance_df.sort_values('abs_importance', ascending=True),
                 x='importance', y='feature', orientation='h',
                 title="Feature Importance: What Pays the Debts?",
                 color_discrete_sequence=[LANNISTER_GOLD], # Gold bars
                 labels={'importance': 'Influence (Coefficient/Value)', 'feature': 'Factor'})
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        plot_bgcolor='rgba(0,0,0,0)', # Transparent plot background
        paper_bgcolor='rgba(0,0,0,0)', # Transparent paper background
        font_color=LANNISTER_TEXT,
        title_font_color=LANNISTER_GOLD,
        yaxis_tickfont_color=LANNISTER_TEXT,
        xaxis_tickfont_color=LANNISTER_TEXT,
        xaxis_title_font_color=LANNISTER_TEXT,
        yaxis_title_font_color=LANNISTER_TEXT,
        legend_font_color=LANNISTER_TEXT
    )
    fig.update_xaxes(gridcolor=LANNISTER_BORDER)
    fig.update_yaxes(gridcolor=LANNISTER_BORDER)
    return fig


def plot_clusters(df, features, cluster_labels, kmeans_model):
    """Plots K-Means clusters using the first two selected features."""
    if len(features) < 2: st.warning("Need at least two features selected to plot clusters."); return None

    cluster_labels_str = cluster_labels.astype(str)
    # Use a color sequence that contrasts well with the dark theme
    color_sequence = px.colors.qualitative.Pastel # Example: Pastel colors might stand out

    fig = px.scatter(df, x=features[0], y=features[1], color=cluster_labels_str,
                     color_discrete_sequence=color_sequence,
                     title=f'Clustering the Realm (K={kmeans_model.n_clusters}) - {features[0]} vs {features[1]}',
                     labels={features[0]: features[0], features[1]: features[1], 'color': 'House (Cluster)'})

    # Add centroids
    if hasattr(kmeans_model, 'cluster_centers_') and 'scaled_feature_names_kmeans' in st.session_state:
        centroids = kmeans_model.cluster_centers_
        scaled_feature_names = st.session_state.scaled_feature_names_kmeans
        try:
            idx1 = scaled_feature_names.index(features[0])
            idx2 = scaled_feature_names.index(features[1])
            fig.add_trace(go.Scatter(x=centroids[:, idx1], y=centroids[:, idx2], mode='markers',
                                     marker=dict(color=LANNISTER_GOLD, size=18, symbol='star', line=dict(width=1, color=LANNISTER_CRIMSON)),
                                     name='Centers of Power'))
        except ValueError: st.warning(f"Cannot plot centroids: Features '{features[0]}' or '{features[1]}' not found in scaled data.")
        except Exception as e: st.warning(f"Could not plot centroids: {e}")
    else: st.warning("Cannot plot centroids: Scaled feature names or centroids missing.")

    fig.update_layout(
        legend_title_text='House (Cluster)',
        plot_bgcolor='rgba(0,0,0,0)', # Transparent plot background
        paper_bgcolor='rgba(0,0,0,0)', # Transparent paper background
        font_color=LANNISTER_TEXT,
        title_font_color=LANNISTER_GOLD,
        xaxis_tickfont_color=LANNISTER_TEXT,
        yaxis_tickfont_color=LANNISTER_TEXT,
        xaxis_title_font_color=LANNISTER_TEXT,
        yaxis_title_font_color=LANNISTER_TEXT,
        legend_font_color=LANNISTER_TEXT,
        xaxis_gridcolor=LANNISTER_BORDER,
        yaxis_gridcolor=LANNISTER_BORDER
        )
    return fig


# --- Initialize Session State ---
def init_session_state():
    """Initializes all required session state variables if they don't exist."""
    defaults = {
        'app_started': False, # For welcome screen
        'data': None, 'data_source': None, 'selected_model_type': None,
        'preprocessed_data': None, 'features': None, 'target': None,
        'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None,
        'model': None, 'predictions': None, 'metrics': None,
        'feature_names': None, 'scaler': None, 'X_scaled_kmeans': None,
        'cluster_labels': None, 'scaled_feature_names_kmeans': None,
        'model_feature_names': None, 'pred_proba': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# --- Reset Function ---
def reset_downstream_state(level='data'):
    """Resets session state variables downstream from a certain level."""
    levels = ['data', 'model_selection', 'preprocessing', 'feature_selection', 'split_scale', 'train', 'evaluate']
    keys_to_reset = []
    try:
        start_index = levels.index(level)
        keys_map = {
            'data': ['preprocessed_data', 'features', 'target', 'X_train', 'X_test', 'y_train', 'y_test', 'model', 'predictions', 'metrics', 'feature_names', 'scaler', 'X_scaled_kmeans', 'cluster_labels', 'scaled_feature_names_kmeans', 'model_feature_names', 'pred_proba'],
            'model_selection': ['preprocessed_data', 'features', 'target', 'X_train', 'X_test', 'y_train', 'y_test', 'model', 'predictions', 'metrics', 'feature_names', 'scaler', 'X_scaled_kmeans', 'cluster_labels', 'scaled_feature_names_kmeans', 'model_feature_names', 'pred_proba'],
            'preprocessing': ['features', 'target', 'X_train', 'X_test', 'y_train', 'y_test', 'model', 'predictions', 'metrics', 'feature_names', 'scaler', 'X_scaled_kmeans', 'cluster_labels', 'scaled_feature_names_kmeans', 'model_feature_names', 'pred_proba'],
            'feature_selection': ['X_train', 'X_test', 'y_train', 'y_test', 'model', 'predictions', 'metrics', 'scaler', 'X_scaled_kmeans', 'cluster_labels', 'scaled_feature_names_kmeans', 'model_feature_names', 'pred_proba'],
            'split_scale': ['model', 'predictions', 'metrics', 'model_feature_names', 'pred_proba'],
            'train': ['predictions', 'metrics', 'pred_proba'],
            'evaluate': []
        }
        for i in range(start_index, len(levels)): keys_to_reset.extend(keys_map.get(levels[i], []))
        keys_to_reset = list(dict.fromkeys(keys_to_reset))
    except ValueError: st.warning(f"Invalid reset level: {level}"); keys_to_reset = []

    reset_count = 0
    for key in keys_to_reset:
        if key in st.session_state and st.session_state[key] is not None:
            st.session_state[key] = None; reset_count += 1
    # if reset_count > 0: st.toast(f"Reset state from {level}.", icon="🧹")


# --- Welcome Screen Logic ---
if not st.session_state.app_started:
    # Use columns to center content better
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        # Apply the welcome-container class for centering
        st.markdown("<div class='welcome-container'>", unsafe_allow_html=True)
        # --- Removed Roaring Lion GIF ---
        st.markdown(f"<h1 style='color:{LANNISTER_GOLD}; font-size: 3rem; text-shadow: 2px 2px 4px #000;'>Hear Me Roar!</h1>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color:{LANNISTER_TEXT};'>Welcome to the Lannister Financial Console</h2>", unsafe_allow_html=True)
        st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbDN1eXlqZzRzN3c1b2w2aXl1MyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/L1JjHInX78b5e/giphy.gif", caption="A Lannister always pays his debts... and analyzes returns.") # Tyrion GIF
        st.markdown("---")
        st.markdown(f"<p style='color:{LANNISTER_TEXT}; font-size: 1.1rem;'>Manage your assets with the precision of Casterly Rock's vaults. Analyze market trends, predict outcomes, and ensure your coffers remain full.</p>", unsafe_allow_html=True)
        if st.button("Enter the Console", key="enter_app"):
            st.session_state.app_started = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# --- Main Application Logic ---
elif st.session_state.app_started:

    # --- Sidebar ---
    with st.sidebar:
        # Lannister Lion Sigil
        # --- Replaced use_column_width with use_container_width ---
        st.image("https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbDN1eXlqZzRzN3c1b2w2aXl1MyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3oEjI1erPMTMBFmNHi/giphy.gif", use_container_width=True)
        st.markdown(f"<h2 style='color:{LANNISTER_GOLD}; text-align: center;'>Lannister Console</h2>", unsafe_allow_html=True)
        st.divider()

        # --- 1. Load Data ---
        st.subheader("1. Load Treasury Data")
        data_load_option = st.radio("Source:", ("Upload Ledger (CSV)", "Scry Market (Yahoo)"),
                                    key="data_source_radio", index=0 if st.session_state.data_source != 'yahoo' else 1,
                                    horizontal=True)

        if data_load_option == "Upload Ledger (CSV)":
            uploaded_file = st.file_uploader("Upload Ledger:", type="csv", key="csv_uploader", label_visibility="collapsed")
            if uploaded_file is not None:
                 st.write(f"Ledger: `{uploaded_file.name}`")
                 if st.button("Load Ledger", key="load_csv_btn", use_container_width=True):
                    try:
                        df = pd.read_csv(uploaded_file)
                        if df.empty: st.error("This ledger is empty!")
                        else:
                             st.session_state.data = df; st.session_state.data_source = 'upload'
                             reset_downstream_state('data')
                             st.success("Ledger loaded successfully!"); st.rerun()
                    except Exception as e: st.error(f"Error loading ledger: {e}"); st.exception(e)

        elif data_load_option == "Scry Market (Yahoo)":
            ticker = st.text_input("Market Ticker(s) (e.g., GOLD, WINE)", "GOLD", key="yahoo_ticker_input") # Themed example
            col1, col2 = st.columns(2)
            with col1: period = st.selectbox("Period:", ['1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'], index=5, key="yahoo_period_select")
            with col2: interval = st.selectbox("Interval:", ['1d','5d','1wk','1mo','3mo'], index=0, key="yahoo_interval_select") # Removed smaller intervals for theme

            if st.button("Scry the Market", key="fetch_yahoo_btn", use_container_width=True):
                if ticker:
                    if not all(c.isalnum() or c in ['-', '.', '^'] for c in ticker.replace(' ', '')): st.warning("Invalid ticker format.")
                    else:
                        try:
                            tickers = ticker.upper().split(); st.info(f"Scrying market for: {', '.join(tickers)}...")
                            data = yf.download(tickers, period=period, interval=interval, progress=False)
                            if isinstance(data, dict): st.error("Market scrying failed (partial data). Check tickers."); st.write(data)
                            elif data.empty: st.warning(f"No market data found for {ticker} ({period}, {interval}).")
                            else:
                                data_df = data.copy()
                                if isinstance(data_df.columns, pd.MultiIndex):
                                     if data_df.index.name in ['Date', 'Datetime']: data_df = data_df.reset_index()
                                     data_df.columns = ["_".join(map(str, col)).strip().replace(' ', '_').replace('<', '').replace('>', '') for col in data_df.columns.values]
                                     date_col_options = [c for c in data_df.columns if c.lower().startswith('date') or c.lower().startswith('datetime')]
                                     date_col = date_col_options[0] if date_col_options else None
                                     close_cols = [col for col in data_df.columns if 'Close' in col and col != date_col]
                                     if close_cols: data_df = data_df.dropna(subset=close_cols, how='all')
                                elif data_df.index.name in ['Date', 'Datetime']: data_df = data_df.reset_index()
                                date_col_to_convert = next((col for col in data_df.columns if col.lower() in ['date', 'datetime']), None)
                                if date_col_to_convert and not pd.api.types.is_datetime64_any_dtype(data_df[date_col_to_convert]):
                                     try: data_df[date_col_to_convert] = pd.to_datetime(data_df[date_col_to_convert])
                                     except Exception as date_e: st.warning(f"Could not parse date column '{date_col_to_convert}': {date_e}")
                                if data_df.empty: st.warning("Market data empty after processing."); st.session_state.data = None; reset_downstream_state('data')
                                else:
                                     st.session_state.data = data_df; st.session_state.data_source = 'yahoo'
                                     reset_downstream_state('data'); st.success(f"Market data for {ticker} acquired!"); st.rerun()
                        except Exception as e: st.error(f"Error scrying market for '{ticker}': {e}"); st.exception(e)
                else: st.warning("Enter market ticker(s).")
        st.divider()

        # --- 2. Select Model ---
        st.subheader("2. Choose Strategy")
        if st.session_state.data is not None:
            model_options = ["Linear Regression", "Logistic Regression", "K-Means Clustering"]
            model_names = ["Predict Value (Regression)", "Predict Outcome (Classification)", "Group Assets (Clustering)"] # Themed names
            current_model_index = model_options.index(st.session_state.selected_model_type) if st.session_state.selected_model_type in model_options else 0

            model_display_name = st.selectbox(
                "Select Analytical Strategy:", model_names, index=current_model_index, key="model_select",
                help="Choose the method for analyzing the financial data." )
            # Map display name back to actual model type
            model_type = model_options[model_names.index(model_display_name)]

            if model_type != st.session_state.selected_model_type:
                 st.session_state.selected_model_type = model_type
                 reset_downstream_state('model_selection')
                 st.toast(f"Strategy set: {model_display_name}", icon="📜"); st.rerun()
        else: st.info("Load treasury data first.")
        st.divider()

    # --- Main Application Area ---
    st.title(f"🦁 Lannister Financial Analysis Console 🦁")

    if st.session_state.data is None: # Should not happen if app_started is True, but safety check
        st.error("Error: No data loaded. Please use the sidebar.")
    else:
        # --- Step 1: Data Overview ---
        st.header("Step 1: Review the Ledgers")
        col1, col2 = st.columns([3, 1])
        with col1: st.dataframe(st.session_state.data.head())
        with col2:
            if st.session_state.data is not None and not st.session_state.data.empty:
                 st.metric("Entries", f"{st.session_state.data.shape[0]:,}")
                 st.metric("Factors", f"{st.session_state.data.shape[1]}")
            else: st.metric("Data Shape", "N/A")
            if st.session_state.data_source: st.metric("Source", st.session_state.data_source.upper())
            else: st.metric("Source", "N/A")
        with st.expander("Examine Ledger Details"):
             if st.session_state.data is not None and not st.session_state.data.empty:
                  st.write("Ledger Tail:"); st.dataframe(st.session_state.data.tail())
                  st.write("Statistical Summary (Numeric Factors):")
                  st.dataframe(st.session_state.data.describe(include=np.number))
             else: st.write("No data loaded.")

        # --- Step 2: Preprocessing ---
        st.header("Step 2: Refine the Data")
        if st.session_state.data is not None and st.session_state.preprocessed_data is None:
            st.info("Cleanse the ledger: Address missing values and encode non-numeric factors.")
            if st.button("Refine Ledger", key="preprocess_btn", use_container_width=True):
                try:
                    with st.spinner("Refining data... A Lannister is meticulous."):
                        data_to_process = st.session_state.data.copy()
                        # --- Missing Values ---
                        st.write("**Missing Values Before Refining:**")
                        missing_vals = data_to_process.isnull().sum(); missing_vals_df = missing_vals[missing_vals > 0].reset_index(name='count').rename(columns={'index':'Column'})
                        if not missing_vals_df.empty: st.dataframe(missing_vals_df)
                        else: st.write("No missing values found.")
                        numeric_cols = data_to_process.select_dtypes(include=np.number).columns; categorical_cols = data_to_process.select_dtypes(exclude=np.number).columns
                        imputed_numeric_cols = []; imputed_categorical_cols = []; potential_datetime_cols = []
                        if not numeric_cols.empty and missing_vals[numeric_cols].any():
                            num_imputer = SimpleImputer(strategy='mean'); data_to_process[numeric_cols] = num_imputer.fit_transform(data_to_process[numeric_cols])
                            imputed_numeric_cols = numeric_cols[missing_vals[numeric_cols] > 0].tolist()
                            if imputed_numeric_cols: st.write(f"Imputed numeric columns (mean): `{', '.join(imputed_numeric_cols)}`")
                        if not categorical_cols.empty:
                             for col in categorical_cols:
                                  if pd.api.types.is_datetime64_any_dtype(data_to_process[col]): potential_datetime_cols.append(col); continue
                                  try:
                                       if pd.to_datetime(data_to_process[col], errors='coerce').notna().mean() > 0.5: potential_datetime_cols.append(col)
                                  except Exception: pass
                             cols_to_impute = categorical_cols.difference(potential_datetime_cols)
                             if not cols_to_impute.empty and missing_vals[cols_to_impute].any():
                                cat_imputer = SimpleImputer(strategy='most_frequent'); data_to_process[cols_to_impute] = cat_imputer.fit_transform(data_to_process[cols_to_impute])
                                imputed_categorical_cols = cols_to_impute[missing_vals[cols_to_impute] > 0].tolist()
                                if imputed_categorical_cols: st.write(f"Imputed categorical columns (mode): `{', '.join(imputed_categorical_cols)}`")
                        st.write("**Missing Values After Refining:**")
                        missing_vals_after = data_to_process.isnull().sum(); missing_vals_after_df = missing_vals_after[missing_vals_after > 0].reset_index(name='count').rename(columns={'index':'Column'})
                        if not missing_vals_after_df.empty: st.dataframe(missing_vals_after_df)
                        else: st.success("No missing values remain.")
                        # --- Encoding ---
                        st.write("**Encoding Factors:**")
                        encoders = {}; encoded_cols = []
                        cols_to_encode = data_to_process.select_dtypes(include=['object', 'category']).columns
                        for col in cols_to_encode:
                             if col in potential_datetime_cols: continue
                             n_unique = data_to_process[col].nunique()
                             if n_unique > 1 and n_unique < 100:
                                  try: le = LabelEncoder(); data_to_process[col] = le.fit_transform(data_to_process[col].astype(str)); encoders[col] = le; encoded_cols.append(col)
                                  except Exception as le_error: st.warning(f"Could not encode '{col}': {le_error}.")
                             elif n_unique >= 100: st.warning(f"Skipping encoding high-cardinality '{col}' ({n_unique})")
                             else: st.warning(f"Skipping encoding low-variance '{col}' ({n_unique})")
                        if encoded_cols: st.write(f"Encoded factors: `{', '.join(encoded_cols)}`")
                        else: st.write("No factors required encoding.")
                        st.session_state.preprocessed_data = data_to_process; st.success("Data refined.")
                        st.dataframe(st.session_state.preprocessed_data.head())
                        reset_downstream_state('preprocessing'); st.rerun()
                except Exception as e: st.error(f"Error refining data: {e}"); st.exception(e)
        elif st.session_state.preprocessed_data is not None:
             st.success("Step 2: Data already refined.")
             with st.expander("Show Refined Data Head"): st.dataframe(st.session_state.preprocessed_data.head())
        elif st.session_state.data is None: st.warning("Load data first.")

        # --- Step 3: Feature Engineering / Selection ---
        st.header("Step 3: Select Factors of Influence")
        if st.session_state.preprocessed_data is not None:
            if st.session_state.features is None:
                st.info("Choose the input factors (X) and the target outcome (y) for the analysis.")
                potential_features = st.session_state.preprocessed_data.columns.tolist()
                datetime_cols = st.session_state.preprocessed_data.select_dtypes(include=['datetime', 'datetime64[ns]', 'timedelta']).columns
                potential_features = [col for col in potential_features if col not in datetime_cols]
                numeric_features = st.session_state.preprocessed_data[potential_features].select_dtypes(include=np.number).columns.tolist()
                # --- Feature Selection ---
                st.write("**Select Input Factors (X)**")
                default_feature_selection = numeric_features if numeric_features else potential_features[:-1] if len(potential_features) > 1 else potential_features
                selected_features = st.multiselect("Select one or more factors:", potential_features, default=default_feature_selection, key="feature_select_ms", help="Inputs for the strategy.")
                # --- Target Selection ---
                target_column = None
                if st.session_state.selected_model_type != "K-Means Clustering":
                    st.write("**Select Target Outcome (y)**")
                    potential_targets = [col for col in potential_features if col not in selected_features]
                    default_target_guess = None; numeric_potential_targets = [t for t in potential_targets if t in numeric_features]
                    common_targets = ['Close', 'Adj Close', 'Volume', 'Target', 'Label', 'Signal', 'Profit', 'Return'] # Added more finance terms
                    for t in common_targets:
                         matching_numeric_targets = [pt for pt in numeric_potential_targets if t.lower() == pt.lower() or pt.lower().endswith(f'_{t.lower()}')]
                         if matching_numeric_targets: default_target_guess = matching_numeric_targets[0]; break
                    if not default_target_guess and numeric_potential_targets: default_target_guess = numeric_potential_targets[0]
                    elif not default_target_guess and potential_targets: default_target_guess = potential_targets[-1]
                    target_index = potential_targets.index(default_target_guess) if default_target_guess in potential_targets else 0
                    target_column = st.selectbox("Select the target outcome:", potential_targets, index=target_index, key="target_select_sb", help="Outcome to predict.")
                    # Logistic Regression Check
                    if st.session_state.selected_model_type == "Logistic Regression" and target_column:
                         target_series = st.session_state.preprocessed_data[target_column]; n_unique = target_series.nunique(); is_numeric = pd.api.types.is_numeric_dtype(target_series)
                         if n_unique != 2: st.warning(f"Target '{target_column}' has {n_unique} values (expected 2).")
                         elif not is_numeric: st.warning(f"Target '{target_column}' is not numeric.")
                         else: st.success(f"Binary numeric target '{target_column}' selected.")
                # --- Confirmation Button ---
                if st.button("Confirm Factors", key="confirm_features_btn", use_container_width=True):
                    valid = True
                    if not selected_features: st.warning("Select at least one input factor."); valid = False
                    if st.session_state.selected_model_type != "K-Means Clustering":
                        if not target_column: st.warning("Select a target outcome."); valid = False
                        elif target_column in selected_features: st.error("Target cannot also be an input factor."); valid = False
                    if st.session_state.selected_model_type == "Logistic Regression" and target_column and valid:
                        target_series = st.session_state.preprocessed_data[target_column]
                        if target_series.nunique() != 2 or not pd.api.types.is_numeric_dtype(target_series): st.error("Invalid target for classification (must be binary numeric)."); valid = False
                    if valid:
                        st.session_state.features = selected_features; st.session_state.target = target_column; st.session_state.feature_names = selected_features
                        st.success("Factors confirmed."); reset_downstream_state('feature_selection'); st.rerun()
            else: # Features already selected
                 st.success("Step 3: Factors of Influence already selected.")
                 st.write("**Input Factors (X):**", f"`{', '.join(st.session_state.features)}`")
                 if st.session_state.target: st.write("**Target Outcome (y):**", f"`{st.session_state.target}`")
        else: st.info("Refine the data (Step 2) first.")

        # --- Step 4: Train/Test Split or Scaling ---
        st.header(f"Step 4: {'Prepare the Battlefield (Split & Scale)' if st.session_state.selected_model_type != 'K-Means Clustering' else 'Standardize Measures (Scale)'}")
        if st.session_state.features:
            step4_done = (st.session_state.X_train is not None) if st.session_state.selected_model_type != "K-Means Clustering" else (st.session_state.X_scaled_kmeans is not None)
            if not step4_done:
                if st.session_state.selected_model_type != "K-Means Clustering":
                    st.info("Divide forces into training and testing groups. Standardize numeric factor measures.")
                    test_size = st.slider("Reserve for Testing (%):", 10, 50, 20, 5, key="test_size_slider", format="%d%%", help="Portion of data held back for final evaluation.") / 100.0
                    random_state = st.number_input("Strategy Seed (Random State):", value=42, key="random_state_split", help="Ensures reproducible division.")
                    if st.button("Divide and Standardize", key="split_scale_btn", use_container_width=True):
                        try:
                            with st.spinner("Dividing forces..."):
                                X = st.session_state.preprocessed_data[st.session_state.features]; y = st.session_state.preprocessed_data[st.session_state.target]
                                numeric_features_in_X = X.select_dtypes(include=np.number).columns; non_numeric_features_in_X = X.select_dtypes(exclude=np.number).columns
                                X_processed = pd.DataFrame(index=X.index)
                                if not numeric_features_in_X.empty:
                                    scaler = StandardScaler(); X_scaled_numeric = scaler.fit_transform(X[numeric_features_in_X]); X_scaled_numeric_df = pd.DataFrame(X_scaled_numeric, index=X.index, columns=numeric_features_in_X)
                                    st.session_state.scaler = scaler; X_processed[numeric_features_in_X] = X_scaled_numeric_df; st.write(f"Standardized factors: `{', '.join(numeric_features_in_X)}`")
                                else: st.write("No numeric factors to standardize.")
                                if not non_numeric_features_in_X.empty: X_processed[non_numeric_features_in_X] = X[non_numeric_features_in_X]; st.write(f"Kept non-numeric factors: `{', '.join(non_numeric_features_in_X)}`")
                                X_processed = X_processed[st.session_state.features]
                                stratify_target = y if st.session_state.selected_model_type == "Logistic Regression" and y.nunique() < 10 else None
                                X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=test_size, random_state=random_state, stratify=stratify_target)
                                st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = X_train, X_test, y_train, y_test
                                st.success("Forces divided and standardized.")
                                st.write(f"Training Forces: X {X_train.shape}, y {y_train.shape}"); st.write(f"Testing Forces: X {X_test.shape}, y {y_test.shape}")
                                labels = ['Training Forces', 'Testing Forces']; sizes = [len(X_train), len(X_test)]
                                fig = px.pie(values=sizes, names=labels, title='Division of Forces', color_discrete_sequence=[LANNISTER_CRIMSON, LANNISTER_GOLD])
                                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=LANNISTER_TEXT, legend_font_color=LANNISTER_TEXT)
                                st.plotly_chart(fig, use_container_width=True); reset_downstream_state('split_scale'); st.rerun()
                        except Exception as e: st.error(f"Error dividing forces: {e}"); st.exception(e)
                else: # K-Means Scaling
                     st.info("Standardize numeric factor measures for clustering.")
                     if st.button("Standardize Measures", key="scale_kmeans_btn", use_container_width=True):
                         try:
                              with st.spinner("Standardizing..."):
                                   X = st.session_state.preprocessed_data[st.session_state.features]; numeric_features_in_X = X.select_dtypes(include=np.number).columns
                                   if not numeric_features_in_X.empty:
                                        scaler = StandardScaler(); X_scaled_numeric = scaler.fit_transform(X[numeric_features_in_X])
                                        st.session_state.X_scaled_kmeans = X_scaled_numeric; st.session_state.scaler = scaler; st.session_state.scaled_feature_names_kmeans = numeric_features_in_X.tolist()
                                        st.success("Measures standardized."); st.write("Standardized Shape:", X_scaled_numeric.shape); st.write(f"Factors standardized: `{', '.join(numeric_features_in_X)}`")
                                   else: st.error("Clustering requires numeric factors. None found."); st.session_state.X_scaled_kmeans = None; st.session_state.scaled_feature_names_kmeans = None
                                   reset_downstream_state('split_scale'); st.rerun()
                         except Exception as e: st.error(f"Error standardizing: {e}"); st.exception(e)
            else: # Step 4 already done
                 st.success(f"Step 4: {'Forces Divided & Standardized' if st.session_state.selected_model_type != 'K-Means Clustering' else 'Measures Standardized'} already completed.")
                 with st.expander("Show Data Shapes"):
                     if st.session_state.selected_model_type != "K-Means Clustering":
                          if st.session_state.X_train is not None: st.write(f"Training Forces: X {st.session_state.X_train.shape}, y {st.session_state.y_train.shape}"); st.write(f"Testing Forces: X {st.session_state.X_test.shape}, y {st.session_state.y_test.shape}")
                          else: st.warning("Split data not found.")
                     else:
                          if st.session_state.X_scaled_kmeans is not None: st.write("Standardized Shape:", st.session_state.X_scaled_kmeans.shape); st.write(f"Factors standardized: `{', '.join(st.session_state.scaled_feature_names_kmeans)}`")
                          else: st.warning("Standardized data not found.")
        else: st.info("Select factors of influence (Step 3) first.")

        # --- Step 5: Model Training ---
        st.header("Step 5: Execute Strategy")
        prerequisites_met = (st.session_state.X_train is not None) if st.session_state.selected_model_type != "K-Means Clustering" else (st.session_state.X_scaled_kmeans is not None)
        if prerequisites_met:
            if st.session_state.model is None:
                st.info(f"Configure parameters and execute the {st.session_state.selected_model_type} strategy.")
                model_params = {}
                with st.container(border=True):
                    st.write(f"**Configure {st.session_state.selected_model_type} Parameters:**")
                    if st.session_state.selected_model_type == "K-Means Clustering":
                        max_k = min(15, st.session_state.X_scaled_kmeans.shape[0] - 1 if st.session_state.X_scaled_kmeans.shape[0] > 1 else 2)
                        n_clusters = st.slider("Number of Houses (K):", min_value=2, max_value=max_k, value=min(3, max_k), key="kmeans_k_slider", help="Number of groups.")
                        model_params['n_clusters'] = n_clusters
                    if st.session_state.selected_model_type == "Logistic Regression":
                        C_reg = st.number_input("Regularization (C):", 0.01, 100.0, 1.0, 0.01, format="%.2f", key='logreg_c', help="Inverse strength; smaller=stronger.")
                        model_params['C'] = C_reg
                if st.button(f"Execute {st.session_state.selected_model_type} Strategy", key="train_model_btn", use_container_width=True):
                    valid_train = True
                    if st.session_state.selected_model_type == "K-Means Clustering" and model_params['n_clusters'] >= st.session_state.X_scaled_kmeans.shape[0]:
                         st.error(f"K ({model_params['n_clusters']}) >= samples ({st.session_state.X_scaled_kmeans.shape[0]}). Choose smaller K."); valid_train = False
                    if valid_train:
                        try:
                            with st.spinner(f"Executing strategy... Gold wins wars."):
                                model = None; model_feature_names_used = []
                                if st.session_state.selected_model_type == "Linear Regression":
                                    model = LinearRegression(); X_train_numeric = st.session_state.X_train.select_dtypes(include=np.number)
                                    if X_train_numeric.empty: st.error("Regression needs numeric factors."); st.stop()
                                    model.fit(X_train_numeric, st.session_state.y_train); model_feature_names_used = X_train_numeric.columns.tolist()
                                elif st.session_state.selected_model_type == "Logistic Regression":
                                    model = LogisticRegression(random_state=42, solver='liblinear', **model_params)
                                    if st.session_state.y_train.nunique() > 2 or not pd.api.types.is_numeric_dtype(st.session_state.y_train): st.error("Classification target must be binary numeric."); st.stop()
                                    X_train_numeric = st.session_state.X_train.select_dtypes(include=np.number)
                                    if X_train_numeric.empty: st.error("Classification needs numeric factors."); st.stop()
                                    model.fit(X_train_numeric, st.session_state.y_train); model_feature_names_used = X_train_numeric.columns.tolist()
                                elif st.session_state.selected_model_type == "K-Means Clustering":
                                    model = KMeans(random_state=42, n_init=10, **model_params)
                                    model.fit(st.session_state.X_scaled_kmeans); model_feature_names_used = st.session_state.scaled_feature_names_kmeans
                                st.session_state.model = model; st.session_state.model_feature_names = model_feature_names_used
                                st.success(f"{st.session_state.selected_model_type} strategy executed!"); reset_downstream_state('train'); st.rerun()
                        except Exception as e: st.error(f"Error executing strategy: {e}"); st.exception(e)
            else: # Model already trained
                st.success(f"Step 5: {st.session_state.selected_model_type} Strategy already executed.")
                with st.expander("Show Executed Strategy Parameters"):
                    try: st.json(st.session_state.model.get_params())
                    except Exception as e: st.warning(f"Could not display parameters: {e}")
        else: # Prerequisites not met
            if st.session_state.preprocessed_data is None: st.info("Refine data (Step 2) first.")
            elif st.session_state.features is None: st.info("Select factors (Step 3) first.")
            elif st.session_state.selected_model_type != "K-Means Clustering" and st.session_state.X_train is None: st.info("Prepare battlefield (Step 4) first.")
            elif st.session_state.selected_model_type == "K-Means Clustering" and st.session_state.X_scaled_kmeans is None: st.info("Standardize measures (Step 4) first.")
            else: st.info("Complete prior steps.")

        # --- Step 6: Evaluation ---
        st.header("Step 6: Assess the Outcome")
        if st.session_state.model is not None:
             if st.session_state.metrics is None:
                st.info(f"Assess the outcome of the {st.session_state.selected_model_type} strategy.")
                if st.button("Assess Outcome", key="evaluate_model_btn", use_container_width=True):
                    try:
                        with st.spinner("Assessing outcome..."):
                            metrics = {}; predictions = None; pred_proba = None; X_test_processed = None
                            if st.session_state.selected_model_type != "K-Means Clustering":
                                 if st.session_state.X_test is None or st.session_state.y_test is None: st.error("Testing forces not found. Run Step 4."); st.stop()
                                 X_test_numeric = st.session_state.X_test.select_dtypes(include=np.number)
                                 if 'model_feature_names' not in st.session_state or not st.session_state.model_feature_names: st.error("Training factors unknown."); st.stop()
                                 model_features = st.session_state.model_feature_names; available_numeric_test = X_test_numeric.columns.tolist()
                                 missing = set(model_features) - set(available_numeric_test)
                                 if missing: st.error(f"Assessment Error: Factors missing from test data: {missing}."); st.stop()
                                 try: X_test_processed = X_test_numeric[model_features]
                                 except KeyError as e: st.error(f"Error aligning test factors: {e}. Required: {model_features}, Available: {available_numeric_test}"); st.stop()
                            # --- Evaluation Logic ---
                            st.write("**Outcome Assessment:**")
                            if st.session_state.selected_model_type == "Linear Regression":
                                predictions = st.session_state.model.predict(X_test_processed); metrics['MSE'] = mean_squared_error(st.session_state.y_test, predictions); metrics['MAE'] = mean_absolute_error(st.session_state.y_test, predictions); metrics['R2 Score'] = r2_score(st.session_state.y_test, predictions)
                                col1, col2, col3 = st.columns(3); col1.metric("MSE", f"{metrics['MSE']:.4f}", help="Mean Squared Error"); col2.metric("MAE", f"{metrics['MAE']:.4f}", help="Mean Absolute Error"); col3.metric("R2 Score", f"{metrics['R2 Score']:.4f}", help="R-squared")
                            elif st.session_state.selected_model_type == "Logistic Regression":
                                predictions = st.session_state.model.predict(X_test_processed)
                                try: pred_proba = st.session_state.model.predict_proba(X_test_processed)
                                except AttributeError: pred_proba = None
                                metrics['Accuracy'] = accuracy_score(st.session_state.y_test, predictions); cm = confusion_matrix(st.session_state.y_test, predictions); metrics['Confusion Matrix'] = cm.tolist()
                                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}", help="Prediction Accuracy")
                                fig_cm = px.imshow(cm, text_auto=True, aspect="auto", labels=dict(x="Predicted", y="True", color="Count"), x=[str(c) for c in st.session_state.model.classes_], y=[str(c) for c in st.session_state.model.classes_], title="Confusion Matrix", color_continuous_scale=[[0, LANNISTER_DARK], [1, LANNISTER_CRIMSON]]) # Themed colorscale
                                fig_cm.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=LANNISTER_TEXT, title_font_color=LANNISTER_GOLD, xaxis_tickfont_color=LANNISTER_TEXT, yaxis_tickfont_color=LANNISTER_TEXT)
                                st.plotly_chart(fig_cm, use_container_width=True)
                            elif st.session_state.selected_model_type == "K-Means Clustering":
                                labels = st.session_state.model.labels_; st.session_state.cluster_labels = labels; n_labels = len(np.unique(labels)); n_samples = st.session_state.X_scaled_kmeans.shape[0]
                                if n_labels > 1 and n_samples > 1: metrics['Silhouette Score'] = silhouette_score(st.session_state.X_scaled_kmeans, labels)
                                else: metrics['Silhouette Score'] = None; st.warning(f"Silhouette Score N/A ({n_labels} labels, {n_samples} samples).")
                                metrics['Inertia (WCSS)'] = st.session_state.model.inertia_
                                col1, col2 = st.columns(2); ss_val = metrics.get('Silhouette Score'); col1.metric("Silhouette Score", f"{ss_val:.4f}" if ss_val is not None else "N/A", help="Cluster separation (-1 to 1)"); col2.metric("Inertia (WCSS)", f"{metrics['Inertia (WCSS)']:.2f}", help="Cluster compactness (Lower is better)")
                            st.session_state.predictions = predictions; st.session_state.pred_proba = pred_proba; st.session_state.metrics = metrics
                            st.success("Outcome assessed."); reset_downstream_state('evaluate'); st.rerun()
                    except Exception as e: st.error(f"Error assessing outcome: {e}"); st.exception(e)
             else: # Evaluation already done
                 st.success("Step 6: Outcome already assessed.")
                 with st.expander("Show Outcome Assessment", expanded=True):
                     metrics = st.session_state.metrics
                     if st.session_state.selected_model_type == "Linear Regression": col1, col2, col3 = st.columns(3); col1.metric("MSE", f"{metrics.get('MSE', 'N/A'):.4f}"); col2.metric("MAE", f"{metrics.get('MAE', 'N/A'):.4f}"); col3.metric("R2 Score", f"{metrics.get('R2 Score', 'N/A'):.4f}")
                     elif st.session_state.selected_model_type == "Logistic Regression":
                          st.metric("Accuracy", f"{metrics.get('Accuracy', 'N/A'):.4f}")
                          cm_list = metrics.get('Confusion Matrix')
                          if cm_list and hasattr(st.session_state.model, 'classes_'):
                               cm = np.array(cm_list); fig_cm = px.imshow(cm, text_auto=True, aspect="auto", labels=dict(x="Predicted", y="True", color="Count"), x=[str(c) for c in st.session_state.model.classes_], y=[str(c) for c in st.session_state.model.classes_], title="Confusion Matrix", color_continuous_scale=[[0, LANNISTER_DARK], [1, LANNISTER_CRIMSON]])
                               fig_cm.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=LANNISTER_TEXT, title_font_color=LANNISTER_GOLD, xaxis_tickfont_color=LANNISTER_TEXT, yaxis_tickfont_color=LANNISTER_TEXT); st.plotly_chart(fig_cm, use_container_width=True)
                     elif st.session_state.selected_model_type == "K-Means Clustering":
                          col1, col2 = st.columns(2); ss = metrics.get('Silhouette Score'); col1.metric("Silhouette Score", f"{ss:.4f}" if ss is not None else "N/A"); col2.metric("Inertia (WCSS)", f"{metrics.get('Inertia (WCSS)', 'N/A'):.2f}")
        else: st.info("Execute a strategy (Step 5) first.")

        # --- Step 7: Results Visualization & Download ---
        st.header("Step 7: Visualize & Claim Spoils")
        if st.session_state.metrics is not None:
            st.info("Visualize the strategy's performance and download the results or the strategy itself.")
            st.write("**Visualizations:**")
            try:
                # Supervised Visualizations
                if st.session_state.selected_model_type != "K-Means Clustering":
                     if st.session_state.predictions is not None and st.session_state.y_test is not None:
                         # Feature Importance
                         if 'model_feature_names' in st.session_state and st.session_state.model_feature_names and st.session_state.model:
                             fig_imp = plot_feature_importance(st.session_state.model, st.session_state.model_feature_names)
                             if fig_imp: st.plotly_chart(fig_imp, use_container_width=True)
                         else: st.info("Feature importance cannot be plotted.")
                         # Specific plots
                         if st.session_state.selected_model_type == "Linear Regression":
                             results_df = pd.DataFrame({'Actual': st.session_state.y_test, 'Predicted': st.session_state.predictions}); fig_pred = px.scatter(results_df, x='Actual', y='Predicted', title='Actual vs. Predicted Wealth', labels={'Actual': 'Actual Value', 'Predicted': 'Predicted Value'}, trendline='ols', trendline_color_override=LANNISTER_GOLD, hover_data=results_df.columns)
                             min_val = min(results_df['Actual'].min(), results_df['Predicted'].min()); max_val = max(results_df['Actual'].max(), results_df['Predicted'].max()); fig_pred.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color='#AAAAAA', dash='dash'))
                             fig_pred.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=LANNISTER_TEXT, title_font_color=LANNISTER_GOLD, xaxis_tickfont_color=LANNISTER_TEXT, yaxis_tickfont_color=LANNISTER_TEXT, xaxis_gridcolor=LANNISTER_BORDER, yaxis_gridcolor=LANNISTER_BORDER); st.plotly_chart(fig_pred, use_container_width=True)
                         elif st.session_state.selected_model_type == "Logistic Regression": st.info("Confusion Matrix shown during assessment (Step 6).")
                     else: st.warning("Predictions/test data unavailable for visualization.")
                # K-Means Visualization
                elif st.session_state.selected_model_type == "K-Means Clustering":
                     if 'cluster_labels' in st.session_state and st.session_state.cluster_labels is not None and st.session_state.features:
                          df_plot = st.session_state.preprocessed_data.copy()
                          if len(df_plot) == len(st.session_state.cluster_labels):
                               df_plot['Cluster'] = st.session_state.cluster_labels.astype(str)
                               st.write("**Select factors for Cluster Plot:**"); numeric_plot_features = df_plot.select_dtypes(include=np.number).columns.tolist()
                               if 'Cluster' in numeric_plot_features: numeric_plot_features.remove('Cluster')
                               if len(numeric_plot_features) >= 2:
                                    col1, col2 = st.columns(2)
                                    with col1: x_axis = st.selectbox("X-axis Factor:", numeric_plot_features, index=0, key="kmeans_plot_x")
                                    with col2: y_options = [f for f in numeric_plot_features if f != x_axis]; y_index = 0 if not y_options else numeric_plot_features.index(y_options[0]); y_axis = st.selectbox("Y-axis Factor:", numeric_plot_features, index=y_index, key="kmeans_plot_y")
                                    if x_axis and y_axis and x_axis != y_axis:
                                         fig_cluster = plot_clusters(df_plot, [x_axis, y_axis], df_plot['Cluster'], st.session_state.model)
                                         if fig_cluster: st.plotly_chart(fig_cluster, use_container_width=True)
                                    else: st.warning("Select two different numeric factors.")
                               else: st.warning("Need >= 2 numeric factors for 2D plot.")
                          else: st.warning(f"Data length vs cluster labels mismatch.")
                     else: st.warning("Cluster labels or features unavailable.")
            except Exception as e: st.error(f"Error generating visualizations: {e}"); st.exception(e)

            # --- Download Options ---
            st.write("**Claim Spoils (Downloads):**")
            col1, col2 = st.columns(2)
            with col1:
                 if st.session_state.model:
                      model_filename = f"lannister_{st.session_state.selected_model_type.replace(' ', '_').lower()}_strategy.joblib"
                      model_bundle = {'model': st.session_state.model}
                      if st.session_state.scaler: model_bundle['scaler'] = st.session_state.scaler
                      if st.session_state.model_feature_names: model_bundle['feature_names'] = st.session_state.model_feature_names
                      download_file(model_bundle, model_filename, "Strategy Bundle", file_format='joblib')
                 else: st.info("No strategy to download.")
            with col2:
                 results_df_download = None; results_filename = "outcome.csv"; results_label = "Outcome"
                 if st.session_state.predictions is not None and st.session_state.selected_model_type != "K-Means Clustering":
                     results_df_download = st.session_state.X_test.copy()
                     try: results_df_download['Actual_Target'] = st.session_state.y_test.values
                     except ValueError: st.warning("Index mismatch, cannot add Actual_Target.")
                     results_df_download['Predicted_Target'] = st.session_state.predictions
                     if st.session_state.selected_model_type == "Logistic Regression" and st.session_state.pred_proba is not None:
                          if hasattr(st.session_state.model, 'classes_'):
                               for i, class_label in enumerate(st.session_state.model.classes_): results_df_download[f'Probability_{class_label}'] = st.session_state.pred_proba[:, i]
                          else:
                               for i in range(st.session_state.pred_proba.shape[1]): results_df_download[f'Probability_Class_{i}'] = st.session_state.pred_proba[:, i]
                     results_filename = "test_predictions.csv"; results_label = "Test Predictions"
                 elif 'cluster_labels' in st.session_state and st.session_state.cluster_labels is not None and st.session_state.selected_model_type == "K-Means Clustering":
                      results_df_download = st.session_state.preprocessed_data.copy()
                      if len(results_df_download) == len(st.session_state.cluster_labels):
                           results_df_download['Cluster'] = st.session_state.cluster_labels; results_filename = "clustered_data.csv"; results_label = "Clustered Data"
                      else: st.warning("Cluster label alignment error."); results_df_download = None
                 if results_df_download is not None: download_file(results_df_download, results_filename, results_label, file_format='csv')
                 else: st.info("No results available to download.")

            st.success("Analysis complete! The realm profits.")
            # Concluding Lannister GIF
            st.image("https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExbWE5MHYxMHVhczAyMjZlNTR1ZjA0amN1eXJpNG1kMTEydjRwZmZwYiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/2wYYlHuEw1UcsJYgAA/giphy.gif", caption="Always victorious.")

        elif st.session_state.model is not None: st.info("Assess the outcome (Step 6) first.")
        else: st.info("Execute a strategy (Step 5) and assess the outcome (Step 6) first.")

    # --- Footer ---
    st.markdown("---")
    st.markdown(f"<p style='text-align: center; color: {LANNISTER_GOLD};'>*A Lannister Always Pays His Debts*</p>", unsafe_allow_html=True)
