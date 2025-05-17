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
import joblib # For saving/loading models
import io # For downloading files
import warnings

# Suppress specific warnings if needed (optional)
# warnings.filterwarnings("ignore", category=FutureWarning)

# --- Theme Options ---
theme_options = {
    "Game of Thrones": {
        "colors": {
            "primary": "#8B0000", # Lannister Crimson
            "accent": "#FFD700",  # Lannister Gold
            "bg": "#1A1A1A",      # Lannister Dark
            "text": "#EAEAEA",    # Lannister Text
            "header": "#FFFFFF",  # Lannister Text Header (brighter)
            "sidebar": "#2B2B2B", # Lannister Sidebar
            "border": "#444444",  # Lannister Border
            "input_bg": "#303030", # For input fields in main area
            "sidebar_input_bg": "#383838", # For input fields in sidebar
            "button_text": "#FFD700", # Explicit button text (accent)
            "button_hover_bg": "#FFD700",
            "button_hover_text": "#8B0000",
        },
        "font": "Cinzel",
        "quotes": {
            "welcome_title": "Hear Me Roar!",
            "welcome_sub": "Welcome to the Lannister Financial Console",
            "quote": "A Lannister always pays his debts...",
            "complete": "Analysis complete! The realm profits.",
            "download_label": "Hear Me Roar! (Download {label})"
        },
        "gifs": {
            "welcome": "https://media.giphy.com/media/L1JjHInX78b5e/giphy.gif",
            "footer": "https://media1.giphy.com/media/2wYYlHuEw1UcsJYgAA/giphy.gif",
            "sidebar": "https://media.giphy.com/media/3oEjI1erPMTMBFmNHi/giphy.gif"
        },
        "icon": "🦁"
    },
    "WWE theme": {
        "colors": {
            "primary": "#000000",
            "accent": "#00FF00",
            "bg": "#111111",
            "text": "#FFFFFF",
            "header": "#00FF00",
            "sidebar": "#222222",
            "border": "#00FF00",
            "input_bg": "#252525",
            "sidebar_input_bg": "#303030",
            "button_text": "#00FF00", # Accent
            "button_hover_bg": "#00FF00",
            "button_hover_text": "#000000",
        },
        "font": "Bangers",
        "quotes": {
            "welcome_title": "Here Comes the Money!",
            "welcome_sub": "Welcome to the SmackDown Stock Show",
            "quote": "If ya smell... what The Rock is cookin' - it's profits!",
            "complete": "Analysis complete! Money talks.",
            "download_label": "Download the Smackdown: {label}"
        },
        "gifs": {
            "welcome": "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExZGtrbDBvd3Rkbmo1ZGMxcTZ6Nmo5c2U4NjdxdHhyeWNvcnl6MTg5ZCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/UiBmJv6Hh6FfW/giphy.gif",
            "footer": "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExZXQ0b2ZmdXhtYTB0Y3B2dHY2ZXZwYm8za2VnazJ1aXNmNGxnaHpodCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/oOl70OBTe6t6IKLIhx/giphy.gif",
            "sidebar": "https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExMW90eTBmbHJneDV3OThuNGhqcW9pMHA0MHBkMGhna2lqbjh2YTcycSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/XfOzScbB2XhzVVzKeI/giphy.gif"
        },
        "icon": "💰"
    },
    "Spongebob": {
        "colors": {
            "primary": "#fce303", # Yellow
            "accent": "#ff6f61",  # Coral Pink
            "bg": "#fdf5e6",      # Seashell
            "text": "#2e2e2e",    # Dark Gray/Black
            "header": "#ff6f61",  # Coral Pink
            "sidebar": "#fffacd", # Lemon Chiffon
            "border": "#ffdab9",  # Peach Puff
            "input_bg": "#FFFFFF",
            "sidebar_input_bg": "#f0e6bb", # Darker Lemon Chiffon
            "button_text": "#2e2e2e",
            "button_hover_bg": "#ff6f61",
            "button_hover_text": "#fdf5e6",
        },
        "font": "Comic Neue",
        "quotes": {
            "welcome_title": "I'm Ready!",
            "welcome_sub": "Welcome to Bikini Bottom Analytics",
            "quote": "Is mayonnaise a financial instrument?",
            "complete": "Analysis complete! I'm ready, I'm ready!",
            "download_label": "Order Up! Download {label}"
        },
        "gifs": {
            "welcome": "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExeTFzc3Q1a2Rtajd2bnF3M2sydWF2aDZrMGZqeW82cnBra2ZzMWpzeCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/LdOyjZ7io5Msw/giphy.gif",
            "footer": "https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExeThiOG5lamh6cjgxYXhqMzBxZTlmczY1cDRncW8wd3JmcWQ0eWJtYiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/6tHy8UAbv3zgs/giphy.gif",
            "sidebar": "https://media1.giphy.com/media/v1.Y2lkPTc5MGI3NjExaW4xaXUyY2JjNDdsOHRva2gwdnJwMWlsNGgzdmJlanVmajZjbzk3aiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/yYrYPXatpCMiA/giphy.gif"
        },
        "icon": "🍍"
    },
    "Breaking Bad": {
        "colors": {
            "primary": "#003300", # Dark Green
            "accent": "#39FF14",  # Bright Green
            "bg": "#121212",      # Very Dark Gray
            "text": "#e0e0e0",    # Light Gray
            "header": "#39FF14",  # Bright Green
            "sidebar": "#1f1f1f", # Darker Gray
            "border": "#2e8b57",  # Sea Green
            "input_bg": "#252525",
            "sidebar_input_bg": "#2a2a2a",
            "button_text": "#39FF14", # Accent
            "button_hover_bg": "#39FF14",
            "button_hover_text": "#003300",
        },
        "font": "Share Tech Mono",
        "quotes": {
            "welcome_title": "I Am the Danger.",
            "welcome_sub": "Welcome to Heisenberg's Financial Lab",
            "quote": "Say my name... Profit.",
            "complete": "Analysis complete. You're goddamn right.",
            "download_label": "Download Lab Results: {label}"
        },
        "gifs": {
            "welcome": "https://media0.giphy.com/media/v1.Y2lkPTc5MGI3NjExaW44eGV2anE2YmZrMzB6eHdhcmRpOXQ5ZXYzczZ5bzA1MGx4YzliZiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/WQJ2DORvilpEk/giphy.gif",
            "footer": "https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExazFrY3ppcm1iNTllcTlyYm51ZWVpNDh0cnJid3FiMnh4bXFwaHdnbCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/xT0GqssRweIhlz209i/giphy.gif",
            "sidebar": "https://giphy.com/gifs/breaking-bad-wink-walter-white-R3S6MfUoKvBVS"
        },
        "icon": "🧪"
    }
}

# --- Initialize session state for selected_theme_name if not exists ---
# This MUST be done BEFORE st.set_page_config if page_icon/title depend on it.
if 'selected_theme_name' not in st.session_state:
    st.session_state.selected_theme_name = list(theme_options.keys())[0] # Default to first theme

# --- Get current theme from session state for st.set_page_config ---
# This theme variable is temporary, will be redefined after sidebar for the main CSS
_initial_theme_for_config = theme_options[st.session_state.selected_theme_name]

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title=f"{st.session_state.selected_theme_name} Financial Analysis",
    page_icon=_initial_theme_for_config["icon"],
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- THEME SELECTION AND GLOBAL THEME VARIABLE SETUP ---
# This block should be executed early, after st.set_page_config.
# The `theme` variable set here will be used by the rest of the app.

# Get the current theme based on session state
theme = theme_options[st.session_state.selected_theme_name]

with st.sidebar:
    # st.write("--- DEBUG: Sidebar - Theme Selector Section ---") # DEBUG
    try:
        # Display sidebar elements that depend on the current theme
        st.image(theme["gifs"]["sidebar"], use_container_width=True, caption="Sidebar Image")
        st.markdown(f"<h2 style='color:{theme['colors']['header']}; text-align: center;'>{st.session_state.selected_theme_name} Console</h2>", unsafe_allow_html=True)
        st.divider()
        # st.write("--- DEBUG: Sidebar image and title rendered. ---") # DEBUG
    except KeyError as e:
        st.error(f"Theme Key Error in sidebar display: {e}. Check theme_options for '{st.session_state.selected_theme_name}'.")
    except Exception as e:
        st.error(f"Error displaying sidebar image/title: {e}")


    # Theme Selector Dropdown
    try:
        current_theme_index = list(theme_options.keys()).index(st.session_state.selected_theme_name)
        # st.write(f"--- DEBUG: Current theme index for selectbox: {current_theme_index} ---") # DEBUG

        new_selected_theme_name = st.selectbox(
            "Choose Theme:",
            options=list(theme_options.keys()),
            index=current_theme_index,
            key="theme_selector_sidebar_v3" # Using a new distinct key
        )
        # st.write(f"--- DEBUG: Selectbox rendered. Selected value: {new_selected_theme_name} ---") # DEBUG

        if new_selected_theme_name != st.session_state.selected_theme_name:
            st.session_state.selected_theme_name = new_selected_theme_name
            # st.write(f"--- DEBUG: Theme changed to {new_selected_theme_name}. Rerunning... ---") # DEBUG
            st.rerun() # Rerun to apply new theme from top
    except Exception as e:
        st.error(f"Error in Theme Selector Dropdown: {e}")
        st.exception(e) # Print full traceback for dropdown error
    # st.write("--- DEBUG: End of Theme Selector in Sidebar ---") # DEBUG


# --- Apply Selected Theme CSS (uses the `theme` variable defined above) ---
GOOGLE_FONT = theme["font"]
FONT_URL = f"https://fonts.googleapis.com/css2?family={GOOGLE_FONT.replace(' ', '+')}:wght@400;700&display=swap"

# Define colors from theme for convenience
PRIMARY_COLOR = theme["colors"]["primary"]
ACCENT_COLOR = theme["colors"]["accent"]
BG_COLOR = theme["colors"]["bg"]
TEXT_COLOR = theme["colors"]["text"]
HEADER_COLOR = theme["colors"]["header"]
SIDEBAR_COLOR = theme["colors"]["sidebar"]
BORDER_COLOR = theme["colors"]["border"]
INPUT_BG_COLOR = theme["colors"].get("input_bg", "#303030")
SIDEBAR_INPUT_BG_COLOR = theme["colors"].get("sidebar_input_bg", "#383838")
BUTTON_TEXT_COLOR = theme["colors"].get("button_text", ACCENT_COLOR)
BUTTON_HOVER_BG = theme["colors"].get("button_hover_bg", ACCENT_COLOR)
BUTTON_HOVER_TEXT = theme["colors"].get("button_hover_text", PRIMARY_COLOR)


st.markdown(f"""
<link href="{FONT_URL}" rel="stylesheet">
<style>
    /* Apply Font Globally */
    html, body, [class*="st-"], button, input, select, textarea {{
        font-family: '{GOOGLE_FONT}', sans-serif !important;
    }}
    .stApp {{ background-color: {BG_COLOR}; color: {TEXT_COLOR}; }}
    [data-testid="stSidebar"] > div:first-child {{ background-color: {SIDEBAR_COLOR}; border-right: 1px solid {BORDER_COLOR}; }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] .stText,
    [data-testid="stSidebar"] .stRadio label span {{ color: {TEXT_COLOR} !important; font-weight: 700; }} /* Adjusted sidebar text color to theme text for better visibility */
    [data-testid="stSidebar"] .stTextInput input, [data-testid="stSidebar"] .stNumberInput input {{
        background-color: {SIDEBAR_INPUT_BG_COLOR} !important; color: {TEXT_COLOR} !important; border: 1px solid {BORDER_COLOR} !important;
    }}
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
        background-color: {SIDEBAR_INPUT_BG_COLOR} !important; color: {TEXT_COLOR} !important; border: 1px solid {BORDER_COLOR} !important;
        padding: 0.5em 0.6em !important; height: auto !important; min-height: 40px !important; line-height: 1.6 !important;
    }}
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div > div {{ /* Inner div for text */
         color: {TEXT_COLOR} !important; overflow: visible !important; white-space: normal !important; height: auto !important;
    }}
    /* Styling for dropdown list items in sidebar */
    div[data-baseweb="popover"] ul li {{
        background-color: {SIDEBAR_INPUT_BG_COLOR} !important; color: {TEXT_COLOR} !important;
    }}
    div[data-baseweb="popover"] ul li:hover {{
        background-color: {PRIMARY_COLOR} !important; color: {ACCENT_COLOR} !important;
    }}
    [data-testid="stSidebar"] .stFileUploader label span {{ color: {TEXT_COLOR} !important; }}
    .stButton>button {{
        background-color: {PRIMARY_COLOR}; color: {BUTTON_TEXT_COLOR}; border: 1px solid {ACCENT_COLOR};
        border-radius: 5px; padding: 10px 22px; transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
        font-weight: 700; font-size: 1.05rem; text-transform: uppercase; letter-spacing: 0.5px;
    }}
    .stButton>button:hover {{
        background-color: {BUTTON_HOVER_BG}; color: {BUTTON_HOVER_TEXT};
        border-color: {PRIMARY_COLOR if BUTTON_HOVER_BG == ACCENT_COLOR else ACCENT_COLOR};
    }}
    .stButton>button:focus {{ box-shadow: 0 0 0 3px {ACCENT_COLOR}66; outline: none; }}
    h1, h2, h3 {{ color: {HEADER_COLOR}; font-weight: 700; text-shadow: 1px 1px 2px #00000033; }}
    h1 {{ border-bottom: 2px solid {PRIMARY_COLOR}; padding-bottom: 0.6rem; text-align: center; }}
    h3 {{ margin-top: 2rem; margin-bottom: 1rem; border-top: 1px solid {BORDER_COLOR}; padding-top: 1rem; color: {ACCENT_COLOR}; }}
    .stDataFrame {{ border: 1px solid {BORDER_COLOR}; border-radius: 0px; background-color: {BG_COLOR}; }}
    .stDataFrame thead th {{ background-color: {PRIMARY_COLOR}; color: {ACCENT_COLOR}; font-weight: 700; text-transform: uppercase; }}
    .stDataFrame tbody td {{ color: {TEXT_COLOR}; background-color: {BG_COLOR}; }}
    .stDataFrame tbody tr:nth-child(even) td {{ background-color: color-mix(in srgb, {BG_COLOR} 90%, {TEXT_COLOR} 10%); }}
    [data-testid="stMetricLabel"] {{ color: color-mix(in srgb, {TEXT_COLOR} 70%, {BG_COLOR} 30%); font-size: 0.95rem; text-transform: uppercase; }}
    [data-testid="stMetricValue"] {{ color: {ACCENT_COLOR}; font-size: 1.3rem; font-weight: 700; }}
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div, .stMultiSelect>div>div {{
        border: 1px solid {BORDER_COLOR} !important; border-radius: 0px; background-color: {INPUT_BG_COLOR} !important; color: {TEXT_COLOR} !important;
    }}
    .stSelectbox>div>div, .stMultiSelect>div>div {{ padding-top: 0.4rem; padding-bottom: 0.4rem; }}
    .stSlider [data-baseweb="slider"] > div:nth-child(2) > div {{ background: {PRIMARY_COLOR}; }}
    .stSlider [data-baseweb="slider"] > div:nth-child(3) {{ background: {ACCENT_COLOR}; border: 2px solid {PRIMARY_COLOR}; }}
    .stAlert {{ border-radius: 0px; border: 1px solid {ACCENT_COLOR}; background-color: color-mix(in srgb, {BG_COLOR} 70%, #000000 30%); }}
    [data-testid="stAlert"] p {{ color: {TEXT_COLOR}; }}
    [data-testid="stAlert"][kind="info"] {{ border-left: 5px solid #4a90e2; }}
    [data-testid="stAlert"][kind="success"] {{ border-left: 5px solid #50e3c2; }}
    [data-testid="stAlert"][kind="warning"] {{ border-left: 5px solid {ACCENT_COLOR}; }}
    [data-testid="stAlert"][kind="error"] {{ border-left: 5px solid {PRIMARY_COLOR}; }}
    .welcome-container {{ display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding-top: 2rem; padding-bottom: 2rem; }}
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---#
def download_file(data, filename, label_key, file_format='csv'):
    global theme
    try:
        buffer = io.BytesIO()
        if file_format == 'csv' and isinstance(data, pd.DataFrame):
            if data.empty: st.warning(f"Cannot download {label_key}: Dataframe is empty."); return
            data.to_csv(buffer, index=False)
            mime = 'text/csv'
        elif file_format == 'joblib':
            if data is None: st.error(f"Cannot download {label_key}: Data is None."); return
            joblib.dump(data, buffer)
            mime = 'application/octet-stream'
        else: st.error(f"Unsupported download format for {label_key}: {file_format}"); return

        buffer.seek(0)
        download_button_label = theme["quotes"]["download_label"].format(label=label_key)
        st.download_button(
            label=download_button_label, data=buffer, file_name=filename, mime=mime,
            key=f"download_{filename.replace('.', '_')}_{label_key.replace(' ', '_')}"
        )
    except Exception as e: st.error(f"Error preparing {label_key} for download: {e}")

def plot_feature_importance(model, feature_names):
    global theme
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
                 title=f"Feature Importance: What Influences the Outcome?",
                 color_discrete_sequence=[theme["colors"]["accent"]],
                 labels={'importance': 'Influence (Coefficient/Value)', 'feature': 'Factor'})
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'}, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color=theme["colors"]["text"], title_font_color=theme["colors"]["header"],
        yaxis_tickfont_color=theme["colors"]["text"], xaxis_tickfont_color=theme["colors"]["text"],
        xaxis_title_font_color=theme["colors"]["text"], yaxis_title_font_color=theme["colors"]["text"],
        legend_font_color=theme["colors"]["text"]
    )
    fig.update_xaxes(gridcolor=theme["colors"]["border"])
    fig.update_yaxes(gridcolor=theme["colors"]["border"])
    return fig

def plot_clusters(df, features, cluster_labels, kmeans_model):
    global theme
    if len(features) < 2: st.warning("Need at least two features selected to plot clusters."); return None

    cluster_labels_str = cluster_labels.astype(str)
    try:
        bg_lum = int(theme["colors"]["bg"][1:3], 16) * 0.299 + \
                 int(theme["colors"]["bg"][3:5], 16) * 0.587 + \
                 int(theme["colors"]["bg"][5:7], 16) * 0.114
        color_sequence = px.colors.qualitative.Pastel if bg_lum < 128 else px.colors.qualitative.Bold
    except: # Fallback if color string is unusual
        color_sequence = px.colors.qualitative.Plotly


    fig = px.scatter(df, x=features[0], y=features[1], color=cluster_labels_str,
                     color_discrete_sequence=color_sequence,
                     title=f'Clustering (K={kmeans_model.n_clusters}) - {features[0]} vs {features[1]}',
                     labels={features[0]: features[0], features[1]: features[1], 'color': 'Group (Cluster)'})

    if hasattr(kmeans_model, 'cluster_centers_') and 'scaled_feature_names_kmeans' in st.session_state and st.session_state.scaled_feature_names_kmeans:
        centroids = kmeans_model.cluster_centers_
        scaled_feature_names = st.session_state.scaled_feature_names_kmeans
        try:
            idx1 = scaled_feature_names.index(features[0])
            idx2 = scaled_feature_names.index(features[1])
            fig.add_trace(go.Scatter(x=centroids[:, idx1], y=centroids[:, idx2], mode='markers',
                                     marker=dict(color=theme["colors"]["accent"], size=18, symbol='star',
                                                 line=dict(width=1, color=theme["colors"]["primary"])),
                                     name='Centers'))
        except ValueError: st.warning(f"Cannot plot centroids: Features '{features[0]}' or '{features[1]}' not found in scaled data used for clustering.")
        except Exception as e: st.warning(f"Could not plot centroids: {e}")

    fig.update_layout(
        legend_title_text='Group (Cluster)', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color=theme["colors"]["text"], title_font_color=theme["colors"]["header"],
        xaxis_tickfont_color=theme["colors"]["text"], yaxis_tickfont_color=theme["colors"]["text"],
        xaxis_title_font_color=theme["colors"]["text"], yaxis_title_font_color=theme["colors"]["text"],
        legend_font_color=theme["colors"]["text"],
        xaxis_gridcolor=theme["colors"]["border"], yaxis_gridcolor=theme["colors"]["border"]
    )
    return fig

# --- Initialize Session State ---
def init_session_state():
    defaults = {
        'app_started': False, 'data': None, 'data_source': None, 'selected_model_type': None,
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

init_session_state() # Call it once at the beginning

# --- Reset Function ---
def reset_downstream_state(level='data'):
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
    for key in keys_to_reset:
        if key in st.session_state and st.session_state[key] is not None:
            st.session_state[key] = None

# --- Welcome Screen Logic ---
if not st.session_state.app_started:
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("<div class='welcome-container'>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='color:{theme['colors']['header']}; font-size: 3rem; text-shadow: 2px 2px 4px #00000080;'>{theme['quotes']['welcome_title']}</h1>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color:{theme['colors']['text']};'>{theme['quotes']['welcome_sub']}</h2>", unsafe_allow_html=True)
        st.image(theme["gifs"]["welcome"], caption=theme["quotes"]["quote"])
        st.markdown("---")
        st.markdown(f"<p style='color:{theme['colors']['text']}; font-size: 1.1rem;'>Manage your assets, analyze market trends, predict outcomes, and ensure your coffers remain full.</p>", unsafe_allow_html=True)
        if st.button("Enter the Console", key="enter_app_btn"): # Changed key
            st.session_state.app_started = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# --- Main Application Logic ---
elif st.session_state.app_started:

    # --- Sidebar Data Loading and Model Selection (after theme selector) ---
    with st.sidebar:
        # Theme selector is already defined above this point
        st.divider() # Divider after the theme selector section

        # --- 1. Load Data ---
        st.subheader("1. Load Treasury Data")
        data_load_option = st.radio("Source:", ("Upload Ledger (CSV)", "Scry Market (Yahoo)"),
                                    key="data_source_radio", index=0 if st.session_state.data_source != 'yahoo' else 1,
                                    horizontal=True)

        if data_load_option == "Upload Ledger (CSV)":
            uploaded_file = st.file_uploader("Upload Ledger:", type="csv", key="csv_uploader_main", label_visibility="collapsed") # Changed key
            if uploaded_file is not None:
                 st.write(f"Ledger: `{uploaded_file.name}`")
                 if st.button("Load Ledger", key="load_csv_main_btn", use_container_width=True): # Changed key
                    try:
                        df = pd.read_csv(uploaded_file)
                        if df.empty: st.error("This ledger is empty!")
                        else:
                             st.session_state.data = df; st.session_state.data_source = 'upload'
                             reset_downstream_state('data')
                             st.success("Ledger loaded successfully!"); st.rerun()
                    except Exception as e: st.error(f"Error loading ledger: {e}"); st.exception(e)

        elif data_load_option == "Scry Market (Yahoo)":
            ticker = st.text_input("Market Ticker(s) (e.g., AAPL, MSFT)", "AAPL", key="yahoo_ticker_main_input") # Changed key
            col_sb1, col_sb2 = st.columns(2)
            with col_sb1: period = st.selectbox("Period:", ['1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'], index=5, key="yahoo_period_main_select") # Changed key
            with col_sb2: interval = st.selectbox("Interval:", ['1d','5d','1wk','1mo','3mo'], index=0, key="yahoo_interval_main_select") # Changed key

            if st.button("Scry the Market", key="fetch_yahoo_main_btn", use_container_width=True): # Changed key
                if ticker:
                    if not all(c.isalnum() or c in ['-', '.', '^', ','] for c in ticker.replace(' ', '')): st.warning("Invalid ticker format.")
                    else:
                        try:
                            tickers_list = [t.strip().upper() for t in ticker.split(',') if t.strip()]
                            if not tickers_list: st.warning("No valid tickers entered."); st.stop()

                            st.info(f"Scrying market for: {', '.join(tickers_list)}...")
                            data = yf.download(tickers_list, period=period, interval=interval, progress=False)

                            if data.empty:
                                st.warning(f"No market data found for {', '.join(tickers_list)} ({period}, {interval}).")
                            else:
                                data_df = data.copy()
                                if isinstance(data_df.columns, pd.MultiIndex):
                                    if len(tickers_list) == 1: data_df.columns = data_df.columns.droplevel(1)
                                    else: data_df.columns = ["_".join(map(str, col)).strip().replace(' ', '_').replace('<', '').replace('>', '') for col in data_df.columns.values]
                                if isinstance(data_df.index, pd.DatetimeIndex): data_df = data_df.reset_index()
                                date_col_to_convert = next((col for col in data_df.columns if col.lower() in ['date', 'datetime']), None)
                                if date_col_to_convert and not pd.api.types.is_datetime64_any_dtype(data_df[date_col_to_convert]):
                                     try: data_df[date_col_to_convert] = pd.to_datetime(data_df[date_col_to_convert])
                                     except Exception as date_e: st.warning(f"Could not parse date column '{date_col_to_convert}': {date_e}")
                                price_cols = [col for col in data_df.columns if any(c.lower() in col.lower() for c in ['open', 'high', 'low', 'close'])]
                                if price_cols: data_df = data_df.dropna(subset=price_cols, how='all')
                                if data_df.empty:
                                    st.warning("Market data empty after processing."); st.session_state.data = None; reset_downstream_state('data')
                                else:
                                     st.session_state.data = data_df; st.session_state.data_source = 'yahoo'
                                     reset_downstream_state('data'); st.success(f"Market data for {', '.join(tickers_list)} acquired!"); st.rerun()
                        except Exception as e: st.error(f"Error scrying market for '{ticker}': {e}"); st.exception(e)
                else: st.warning("Enter market ticker(s).")
        st.divider()

        # --- 2. Select Model ---
        st.subheader("2. Choose Strategy")
        if st.session_state.data is not None:
            model_options_map = {
                "Predict Value (Regression)": "Linear Regression",
                "Predict Outcome (Classification)": "Logistic Regression",
                "Group Assets (Clustering)": "K-Means Clustering"
            }
            model_display_names = list(model_options_map.keys())
            current_model_index = 0
            if st.session_state.selected_model_type:
                try:
                    current_display_name_for_select = [k for k, v in model_options_map.items() if v == st.session_state.selected_model_type][0]
                    current_model_index = model_display_names.index(current_display_name_for_select)
                except (IndexError, ValueError): pass # Keep current_model_index = 0

            selected_display_name = st.selectbox(
                "Select Analytical Strategy:", model_display_names, index=current_model_index, key="model_select_main_display", # Changed key
                help="Choose the method for analyzing the financial data." )
            actual_model_type = model_options_map[selected_display_name]
            if actual_model_type != st.session_state.selected_model_type:
                 st.session_state.selected_model_type = actual_model_type
                 reset_downstream_state('model_selection')
                 st.toast(f"Strategy set: {selected_display_name}", icon="📜"); st.rerun()
        else: st.info("Load treasury data first.")
        st.divider()

    # --- Main Application Area ---
    st.title(f"{theme['icon']} {st.session_state.selected_theme_name} Financial Analysis {theme['icon']}")

    if st.session_state.data is None:
        st.error("Error: No data loaded. Please use the sidebar.")
    else:
        # --- Step 1: Data Overview ---
        st.header("Step 1: Review the Ledgers")
        col1_main, col2_main = st.columns([3, 1])
        with col1_main: st.dataframe(st.session_state.data.head())
        with col2_main:
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
            if st.button("Refine Ledger", key="preprocess_main_btn", use_container_width=True): # Changed key
                try:
                    with st.spinner(f"Refining data... {theme['quotes']['quote'][:20]}..."):
                        data_to_process = st.session_state.data.copy()
                        st.write("**Missing Values Before Refining:**")
                        missing_vals = data_to_process.isnull().sum(); missing_vals_df = missing_vals[missing_vals > 0].reset_index(name='count').rename(columns={'index':'Column'})
                        if not missing_vals_df.empty: st.dataframe(missing_vals_df)
                        else: st.write("No missing values found.")
                        numeric_cols = data_to_process.select_dtypes(include=np.number).columns
                        categorical_cols = data_to_process.select_dtypes(exclude=np.number).columns
                        imputed_numeric_cols = []; imputed_categorical_cols = []; potential_datetime_cols = []
                        if not numeric_cols.empty:
                            cols_with_nan_numeric = numeric_cols[data_to_process[numeric_cols].isnull().any()].tolist()
                            if cols_with_nan_numeric:
                                num_imputer = SimpleImputer(strategy='mean')
                                data_to_process[cols_with_nan_numeric] = num_imputer.fit_transform(data_to_process[cols_with_nan_numeric])
                                imputed_numeric_cols = cols_with_nan_numeric
                                if imputed_numeric_cols: st.write(f"Imputed numeric columns (mean): `{', '.join(imputed_numeric_cols)}`")
                        if not categorical_cols.empty:
                            for col in categorical_cols:
                                try:
                                    if pd.to_datetime(data_to_process[col], errors='coerce').notna().mean() > 0.5 and not pd.api.types.is_datetime64_any_dtype(data_to_process[col]): potential_datetime_cols.append(col)
                                    elif pd.api.types.is_datetime64_any_dtype(data_to_process[col]): potential_datetime_cols.append(col)
                                except Exception: pass
                            cols_to_impute_cat = categorical_cols.difference(potential_datetime_cols)
                            if not cols_to_impute_cat.empty:
                                cols_with_nan_cat = cols_to_impute_cat[data_to_process[cols_to_impute_cat].isnull().any()].tolist()
                                if cols_with_nan_cat:
                                    cat_imputer = SimpleImputer(strategy='most_frequent'); data_to_process[cols_with_nan_cat] = cat_imputer.fit_transform(data_to_process[cols_with_nan_cat])
                                    imputed_categorical_cols = cols_with_nan_cat
                                    if imputed_categorical_cols: st.write(f"Imputed categorical columns (mode): `{', '.join(imputed_categorical_cols)}`")
                        st.write("**Missing Values After Refining:**")
                        missing_vals_after = data_to_process.isnull().sum(); missing_vals_after_df = missing_vals_after[missing_vals_after > 0].reset_index(name='count').rename(columns={'index':'Column'})
                        if not missing_vals_after_df.empty: st.dataframe(missing_vals_after_df)
                        else: st.success("No missing values remain.")
                        st.write("**Encoding Factors:**")
                        encoders = {}; encoded_cols = []
                        cols_to_encode = data_to_process.select_dtypes(include=['object', 'category']).columns.difference(potential_datetime_cols)
                        for col in cols_to_encode:
                             n_unique = data_to_process[col].nunique()
                             if n_unique > 1 and n_unique < 100:
                                  try: le = LabelEncoder(); data_to_process[col] = le.fit_transform(data_to_process[col].astype(str)); encoders[col] = le; encoded_cols.append(col)
                                  except Exception as le_error: st.warning(f"Could not encode '{col}': {le_error}.")
                             elif n_unique >= 100: st.warning(f"Skipping encoding high-cardinality '{col}' ({n_unique})")
                             else: st.warning(f"Skipping encoding low-variance '{col}' ({n_unique})")
                        if encoded_cols: st.write(f"Encoded factors: `{', '.join(encoded_cols)}`")
                        else: st.write("No factors required encoding or met encoding criteria.")
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
                df_processed = st.session_state.preprocessed_data
                potential_features_list = df_processed.columns.tolist()
                datetime_cols_list = df_processed.select_dtypes(include=['datetime', 'datetime64[ns]', 'timedelta']).columns.tolist()
                non_datetime_features_list = [col for col in potential_features_list if col not in datetime_cols_list]
                numeric_features_list = df_processed[non_datetime_features_list].select_dtypes(include=np.number).columns.tolist()
                st.write("**Select Input Factors (X)**")
                default_feature_sel = numeric_features_list if numeric_features_list else non_datetime_features_list[:-1] if len(non_datetime_features_list) > 1 else non_datetime_features_list
                selected_features_ms = st.multiselect("Select one or more factors:", non_datetime_features_list, default=default_feature_sel, key="feature_select_main_ms", help="Inputs for the strategy.") # Changed key
                target_column_sel = None
                if st.session_state.selected_model_type != "K-Means Clustering":
                    st.write("**Select Target Outcome (y)**")
                    potential_targets_list = [col for col in non_datetime_features_list if col not in selected_features_ms]
                    default_target_g = None
                    numeric_potential_targets_list = [t for t in potential_targets_list if t in numeric_features_list]
                    common_targets_kws = ['close', 'adj close', 'volume', 'target', 'label', 'signal', 'profit', 'return', 'price']
                    if numeric_potential_targets_list:
                        for kw_item in common_targets_kws:
                            matches_list = [pt for pt in numeric_potential_targets_list if kw_item in pt.lower()]
                            if matches_list: default_target_g = matches_list[0]; break
                        if not default_target_g: default_target_g = numeric_potential_targets_list[0]
                    elif potential_targets_list: default_target_g = potential_targets_list[-1] if potential_targets_list else None
                    target_idx = potential_targets_list.index(default_target_g) if default_target_g and default_target_g in potential_targets_list else 0
                    target_column_sel = st.selectbox("Select the target outcome:", potential_targets_list, index=target_idx, key="target_select_main_sb", help="Outcome to predict.") # Changed key
                    if st.session_state.selected_model_type == "Logistic Regression" and target_column_sel:
                         target_series_check = df_processed[target_column_sel]; n_unique_check = target_series_check.nunique(); is_numeric_check = pd.api.types.is_numeric_dtype(target_series_check)
                         if n_unique_check < 2 : st.warning(f"Target '{target_column_sel}' has only {n_unique_check} unique value (binary classification needs 2).")
                         elif n_unique_check > 10 and is_numeric_check: st.warning(f"Target '{target_column_sel}' has {n_unique_check} unique numeric values. Consider binning for classification or if this is a regression problem.")
                         elif not is_numeric_check: st.warning(f"Target '{target_column_sel}' is not numeric. Logistic regression usually expects numeric classes (e.g., 0 and 1).")
                         elif n_unique_check == 2 and is_numeric_check: st.success(f"Binary numeric target '{target_column_sel}' selected.")
                if st.button("Confirm Factors", key="confirm_features_main_btn", use_container_width=True): # Changed key
                    valid_confirm = True
                    if not selected_features_ms: st.warning("Select at least one input factor."); valid_confirm = False
                    if st.session_state.selected_model_type != "K-Means Clustering":
                        if not target_column_sel: st.warning("Select a target outcome."); valid_confirm = False
                        elif target_column_sel in selected_features_ms: st.error("Target cannot also be an input factor."); valid_confirm = False
                    if st.session_state.selected_model_type == "Logistic Regression" and target_column_sel and valid_confirm:
                        target_series_val = df_processed[target_column_sel]
                        if not pd.api.types.is_numeric_dtype(target_series_val) and target_series_val.nunique() > 20: st.error("Target for classification is non-numeric with many unique values. Please ensure it's suitable."); valid_confirm = False
                        elif target_series_val.nunique() < 2: st.error("Target for classification must have at least 2 unique values."); valid_confirm = False
                    if valid_confirm:
                        st.session_state.features = selected_features_ms; st.session_state.target = target_column_sel; st.session_state.feature_names = selected_features_ms
                        st.success("Factors confirmed."); reset_downstream_state('feature_selection'); st.rerun()
            else:
                 st.success("Step 3: Factors of Influence already selected.")
                 st.write("**Input Factors (X):**", f"`{', '.join(st.session_state.features)}`")
                 if st.session_state.target: st.write("**Target Outcome (y):**", f"`{st.session_state.target}`")
        else: st.info("Refine the data (Step 2) first.")

        # --- Step 4: Train/Test Split or Scaling ---
        step4_title = "Prepare the Battlefield (Split & Scale)" if st.session_state.selected_model_type != 'K-Means Clustering' else "Standardize Measures (Scale)"
        st.header(f"Step 4: {step4_title}")
        if st.session_state.features:
            step4_is_done = (st.session_state.X_train is not None) if st.session_state.selected_model_type != "K-Means Clustering" else (st.session_state.X_scaled_kmeans is not None)
            if not step4_is_done:
                if st.session_state.selected_model_type != "K-Means Clustering":
                    st.info("Divide forces into training and testing groups. Standardize numeric factor measures.")
                    test_size_val = st.slider("Reserve for Testing (%):", 10, 50, 20, 5, key="test_size_main_slider", format="%d%%", help="Portion of data held back for final evaluation.") / 100.0 # Changed key
                    random_state_val = st.number_input("Strategy Seed (Random State):", value=42, min_value=0, key="random_state_main_split", help="Ensures reproducible division.") # Changed key
                    if st.button("Divide and Standardize", key="split_scale_main_btn", use_container_width=True): # Changed key
                        try:
                            with st.spinner("Dividing forces..."):
                                X_df_split = st.session_state.preprocessed_data[st.session_state.features]; y_series_split = st.session_state.preprocessed_data[st.session_state.target]
                                numeric_feat_in_X = X_df_split.select_dtypes(include=np.number).columns.tolist(); X_proc_split = X_df_split.copy()
                                if numeric_feat_in_X:
                                    scaler_obj = StandardScaler(); X_proc_split[numeric_feat_in_X] = scaler_obj.fit_transform(X_df_split[numeric_feat_in_X])
                                    st.session_state.scaler = scaler_obj; st.write(f"Standardized numeric factors: `{', '.join(numeric_feat_in_X)}`")
                                else: st.write("No numeric factors to standardize in X."); st.session_state.scaler = None
                                st.session_state.model_feature_names = X_proc_split.columns.tolist()
                                stratify_y = y_series_split if st.session_state.selected_model_type == "Logistic Regression" and y_series_split.nunique() < 10 else None
                                X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(X_proc_split, y_series_split, test_size=test_size_val, random_state=random_state_val, stratify=stratify_y)
                                st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = X_train_df, X_test_df, y_train_s, y_test_s
                                st.success("Forces divided and standardized.")
                                st.write(f"Training Forces: X {X_train_df.shape}, y {y_train_s.shape}"); st.write(f"Testing Forces: X {X_test_df.shape}, y {y_test_s.shape}")
                                fig_split_pie = px.pie(values=[len(X_train_df), len(X_test_df)], names=['Training Forces', 'Testing Forces'], title='Division of Forces', color_discrete_sequence=[theme["colors"]["primary"], theme["colors"]["accent"]])
                                fig_split_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=theme["colors"]["text"], legend_font_color=theme["colors"]["text"])
                                st.plotly_chart(fig_split_pie, use_container_width=True); reset_downstream_state('split_scale'); st.rerun()
                        except Exception as e: st.error(f"Error dividing forces: {e}"); st.exception(e)
                else: # K-Means Scaling
                     st.info("Standardize numeric factor measures for clustering.")
                     if st.button("Standardize Measures", key="scale_kmeans_main_btn", use_container_width=True): # Changed key
                         try:
                              with st.spinner("Standardizing..."):
                                   X_df_k = st.session_state.preprocessed_data[st.session_state.features]; numeric_feat_for_k = X_df_k.select_dtypes(include=np.number).columns.tolist()
                                   if not numeric_feat_for_k: st.error("Clustering requires numeric factors. None found in selected features."); st.stop()
                                   scaler_k = StandardScaler(); X_scaled_num_k = scaler_k.fit_transform(X_df_k[numeric_feat_for_k])
                                   st.session_state.X_scaled_kmeans = X_scaled_num_k; st.session_state.scaler = scaler_k; st.session_state.scaled_feature_names_kmeans = numeric_feat_for_k; st.session_state.model_feature_names = numeric_feat_for_k
                                   st.success("Measures standardized."); st.write("Standardized Shape:", X_scaled_num_k.shape); st.write(f"Factors standardized for K-Means: `{', '.join(numeric_feat_for_k)}`")
                                   reset_downstream_state('split_scale'); st.rerun()
                         except Exception as e: st.error(f"Error standardizing: {e}"); st.exception(e)
            else:
                 st.success(f"Step 4: {step4_title} already completed.")
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
        prereqs_met_train = (st.session_state.X_train is not None and st.session_state.y_train is not None) if st.session_state.selected_model_type != "K-Means Clustering" else (st.session_state.X_scaled_kmeans is not None)
        if prereqs_met_train:
            if st.session_state.model is None:
                st.info(f"Configure parameters and execute the {st.session_state.selected_model_type} strategy.")
                model_p = {}
                with st.container(border=True):
                    st.write(f"**Configure {st.session_state.selected_model_type} Parameters:**")
                    if st.session_state.selected_model_type == "K-Means Clustering":
                        max_k_val_slider = st.session_state.X_scaled_kmeans.shape[0] -1 if st.session_state.X_scaled_kmeans.shape[0] > 1 else 1; max_k_val_slider = min(15, max_k_val_slider)
                        if max_k_val_slider < 2: st.warning("Not enough samples for K-Means (K must be >= 2 and < n_samples)."); st.stop()
                        n_clusters_val = st.slider("Number of Groups (K):", min_value=2, max_value=max_k_val_slider, value=min(3, max_k_val_slider), key="kmeans_k_main_slider", help="Number of groups.") # Changed key
                        model_p['n_clusters'] = n_clusters_val
                    if st.session_state.selected_model_type == "Logistic Regression":
                        C_reg_val = st.number_input("Regularization (C):", 0.01, 100.0, 1.0, 0.01, format="%.2f", key='logreg_c_main', help="Inverse strength; smaller=stronger.") # Changed key
                        model_p['C'] = C_reg_val
                if st.button(f"Execute {st.session_state.selected_model_type} Strategy", key="train_model_main_btn", use_container_width=True): # Changed key
                    valid_to_train = True
                    if st.session_state.selected_model_type == "K-Means Clustering" and model_p['n_clusters'] >= st.session_state.X_scaled_kmeans.shape[0]:
                         st.error(f"K ({model_p['n_clusters']}) >= samples ({st.session_state.X_scaled_kmeans.shape[0]}). Choose smaller K."); valid_to_train = False
                    if valid_to_train:
                        try:
                            with st.spinner(f"Executing strategy... {theme['quotes']['quote'][:25]}..."):
                                model_obj_train = None
                                if st.session_state.selected_model_type == "Linear Regression":
                                    model_obj_train = LinearRegression(); model_obj_train.fit(st.session_state.X_train, st.session_state.y_train)
                                elif st.session_state.selected_model_type == "Logistic Regression":
                                    if st.session_state.y_train.nunique() < 2: st.error("Classification target must have at least 2 unique values in training data."); st.stop()
                                    model_obj_train = LogisticRegression(random_state=42, solver='liblinear', **model_p); model_obj_train.fit(st.session_state.X_train, st.session_state.y_train)
                                elif st.session_state.selected_model_type == "K-Means Clustering":
                                    model_obj_train = KMeans(n_clusters=model_p['n_clusters'], random_state=42, n_init='auto'); model_obj_train.fit(st.session_state.X_scaled_kmeans) # Changed n_init
                                st.session_state.model = model_obj_train
                                st.success(f"{st.session_state.selected_model_type} strategy executed!"); reset_downstream_state('train'); st.rerun()
                        except Exception as e: st.error(f"Error executing strategy: {e}"); st.exception(e)
            else:
                st.success(f"Step 5: {st.session_state.selected_model_type} Strategy already executed.")
                with st.expander("Show Executed Strategy Parameters"):
                    try: st.json(st.session_state.model.get_params())
                    except Exception as e: st.warning(f"Could not display parameters: {e}")
        else:
            if st.session_state.preprocessed_data is None: st.info("Refine data (Step 2) first.")
            elif st.session_state.features is None: st.info("Select factors (Step 3) first.")
            elif not prereqs_met_train: st.info(f"{step4_title} (Step 4) first.")
            else: st.info("Complete prior steps.")

        # --- Step 6: Evaluation ---
        st.header("Step 6: Assess the Outcome")
        if st.session_state.model is not None:
             if st.session_state.metrics is None:
                st.info(f"Assess the outcome of the {st.session_state.selected_model_type} strategy.")
                if st.button("Assess Outcome", key="evaluate_model_main_btn", use_container_width=True): # Changed key
                    try:
                        with st.spinner("Assessing outcome..."):
                            metrics_res = {}; preds_res = None; pred_proba_res = None
                            if 'model_feature_names' not in st.session_state or not st.session_state.model_feature_names: st.error("Model feature names not found. Re-run previous steps."); st.stop()
                            model_feat_train = st.session_state.model_feature_names
                            if st.session_state.selected_model_type != "K-Means Clustering":
                                 if st.session_state.X_test is None or st.session_state.y_test is None: st.error("Testing forces not found. Run Step 4."); st.stop()
                                 try: X_test_for_eval = st.session_state.X_test[model_feat_train]
                                 except KeyError as e: st.error(f"Feature mismatch during evaluation. Model trained on {model_feat_train}, but X_test has {st.session_state.X_test.columns.tolist()}. Error: {e}"); st.stop()
                            st.write("**Outcome Assessment:**")
                            if st.session_state.selected_model_type == "Linear Regression":
                                preds_res = st.session_state.model.predict(X_test_for_eval); metrics_res['MSE'] = mean_squared_error(st.session_state.y_test, preds_res); metrics_res['MAE'] = mean_absolute_error(st.session_state.y_test, preds_res); metrics_res['R2 Score'] = r2_score(st.session_state.y_test, preds_res)
                                col_m1, col_m2, col_m3 = st.columns(3); col_m1.metric("MSE", f"{metrics_res['MSE']:.4f}"); col_m2.metric("MAE", f"{metrics_res['MAE']:.4f}"); col_m3.metric("R2 Score", f"{metrics_res['R2 Score']:.4f}")
                            elif st.session_state.selected_model_type == "Logistic Regression":
                                preds_res = st.session_state.model.predict(X_test_for_eval); pred_proba_res = st.session_state.model.predict_proba(X_test_for_eval) if hasattr(st.session_state.model, "predict_proba") else None
                                metrics_res['Accuracy'] = accuracy_score(st.session_state.y_test, preds_res); cm_val = confusion_matrix(st.session_state.y_test, preds_res); metrics_res['Confusion Matrix'] = cm_val.tolist()
                                st.metric("Accuracy", f"{metrics_res['Accuracy']:.4f}")
                                fig_cm_plot = px.imshow(cm_val, text_auto=True, aspect="auto", labels=dict(x="Predicted", y="True", color="Count"), x=[str(c) for c in st.session_state.model.classes_], y=[str(c) for c in st.session_state.model.classes_], title="Confusion Matrix", color_continuous_scale=[[0, theme["colors"]["bg"]], [1, theme["colors"]["primary"]]])
                                fig_cm_plot.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=theme["colors"]["text"], title_font_color=theme["colors"]["header"], xaxis_tickfont_color=theme["colors"]["text"], yaxis_tickfont_color=theme["colors"]["text"])
                                st.plotly_chart(fig_cm_plot, use_container_width=True)
                            elif st.session_state.selected_model_type == "K-Means Clustering":
                                labels_k = st.session_state.model.labels_; st.session_state.cluster_labels = labels_k; n_labels_k = len(np.unique(labels_k)); n_samples_k = st.session_state.X_scaled_kmeans.shape[0]
                                if n_labels_k > 1 and n_labels_k < n_samples_k : metrics_res['Silhouette Score'] = silhouette_score(st.session_state.X_scaled_kmeans, labels_k)
                                else: metrics_res['Silhouette Score'] = None; st.warning(f"Silhouette Score N/A (K={n_labels_k}, n_samples={n_samples_k}).")
                                metrics_res['Inertia (WCSS)'] = st.session_state.model.inertia_
                                col_k1, col_k2 = st.columns(2); ss_val_k = metrics_res.get('Silhouette Score'); col_k1.metric("Silhouette Score", f"{ss_val_k:.4f}" if ss_val_k is not None else "N/A"); col_k2.metric("Inertia (WCSS)", f"{metrics_res['Inertia (WCSS)']:.2f}")
                            st.session_state.predictions = preds_res; st.session_state.pred_proba = pred_proba_res; st.session_state.metrics = metrics_res
                            st.success("Outcome assessed."); reset_downstream_state('evaluate'); st.rerun()
                    except Exception as e: st.error(f"Error assessing outcome: {e}"); st.exception(e)
             else:
                 st.success("Step 6: Outcome already assessed.")
                 with st.expander("Show Outcome Assessment", expanded=True):
                     metrics_d = st.session_state.metrics
                     if st.session_state.selected_model_type == "Linear Regression": col_md1, col_md2, col_md3 = st.columns(3); col_md1.metric("MSE", f"{metrics_d.get('MSE', 'N/A'):.4f}" if isinstance(metrics_d.get('MSE'), (int, float)) else "N/A"); col_md2.metric("MAE", f"{metrics_d.get('MAE', 'N/A'):.4f}" if isinstance(metrics_d.get('MAE'), (int, float)) else "N/A"); col_md3.metric("R2 Score", f"{metrics_d.get('R2 Score', 'N/A'):.4f}" if isinstance(metrics_d.get('R2 Score'), (int, float)) else "N/A")
                     elif st.session_state.selected_model_type == "Logistic Regression":
                          st.metric("Accuracy", f"{metrics_d.get('Accuracy', 'N/A'):.4f}" if isinstance(metrics_d.get('Accuracy'), (int, float)) else "N/A")
                          cm_list_d = metrics_d.get('Confusion Matrix')
                          if cm_list_d and hasattr(st.session_state.model, 'classes_'):
                               cm_np_d = np.array(cm_list_d); fig_cm_d = px.imshow(cm_np_d, text_auto=True, aspect="auto", labels=dict(x="Predicted", y="True", color="Count"), x=[str(c) for c in st.session_state.model.classes_], y=[str(c) for c in st.session_state.model.classes_], title="Confusion Matrix", color_continuous_scale=[[0, theme["colors"]["bg"]], [1, theme["colors"]["primary"]]])
                               fig_cm_d.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=theme["colors"]["text"], title_font_color=theme["colors"]["header"], xaxis_tickfont_color=theme["colors"]["text"], yaxis_tickfont_color=theme["colors"]["text"]); st.plotly_chart(fig_cm_d, use_container_width=True)
                     elif st.session_state.selected_model_type == "K-Means Clustering":
                          col_kd1, col_kd2 = st.columns(2); ss_d = metrics_d.get('Silhouette Score'); col_kd1.metric("Silhouette Score", f"{ss_d:.4f}" if ss_d is not None else "N/A"); col_kd2.metric("Inertia (WCSS)", f"{metrics_d.get('Inertia (WCSS)', 'N/A'):.2f}" if isinstance(metrics_d.get('Inertia (WCSS)'), (int, float)) else "N/A")
        else: st.info("Execute a strategy (Step 5) first.")

        # --- Step 7: Results Visualization & Download ---
        st.header("Step 7: Visualize & Claim Spoils")
        if st.session_state.metrics is not None:
            st.info("Visualize the strategy's performance and download the results or the strategy itself.")
            st.write("**Visualizations:**")
            try:
                if st.session_state.selected_model_type != "K-Means Clustering":
                     if st.session_state.predictions is not None and st.session_state.y_test is not None:
                         if 'model_feature_names' in st.session_state and st.session_state.model_feature_names and st.session_state.model:
                             fig_imp_plot = plot_feature_importance(st.session_state.model, st.session_state.model_feature_names)
                             if fig_imp_plot: st.plotly_chart(fig_imp_plot, use_container_width=True)
                         if st.session_state.selected_model_type == "Linear Regression":
                             results_df_lin = pd.DataFrame({'Actual': st.session_state.y_test, 'Predicted': st.session_state.predictions}); fig_pred_lin = px.scatter(results_df_lin, x='Actual', y='Predicted', title='Actual vs. Predicted Values', labels={'Actual': 'Actual Value', 'Predicted': 'Predicted Value'}, trendline='ols', trendline_color_override=theme["colors"]["accent"], hover_data=results_df_lin.columns)
                             min_val_lin = min(results_df_lin['Actual'].min(), results_df_lin['Predicted'].min()); max_val_lin = max(results_df_lin['Actual'].max(), results_df_lin['Predicted'].max()); fig_pred_lin.add_shape(type='line', x0=min_val_lin, y0=min_val_lin, x1=max_val_lin, y1=max_val_lin, line=dict(color=theme["colors"]["text"], dash='dash'))
                             fig_pred_lin.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=theme["colors"]["text"], title_font_color=theme["colors"]["header"], xaxis_tickfont_color=theme["colors"]["text"], yaxis_tickfont_color=theme["colors"]["text"], xaxis_gridcolor=theme["colors"]["border"], yaxis_gridcolor=theme["colors"]["border"]); st.plotly_chart(fig_pred_lin, use_container_width=True)
                         elif st.session_state.selected_model_type == "Logistic Regression": st.info("Confusion Matrix shown during assessment (Step 6).")
                     else: st.warning("Predictions/test data unavailable for visualization.")
                elif st.session_state.selected_model_type == "K-Means Clustering":
                     if 'cluster_labels' in st.session_state and st.session_state.cluster_labels is not None and 'scaled_feature_names_kmeans' in st.session_state and st.session_state.scaled_feature_names_kmeans:
                          df_plot_k = st.session_state.preprocessed_data.copy()
                          if len(df_plot_k) == len(st.session_state.cluster_labels):
                               df_plot_k['Cluster'] = st.session_state.cluster_labels.astype(str)
                               avail_plot_feat_k = st.session_state.scaled_feature_names_kmeans
                               if len(avail_plot_feat_k) >= 2:
                                    st.write("**Select factors for Cluster Plot (from features used in K-Means):**"); col_p1, col_p2 = st.columns(2)
                                    with col_p1: x_axis_k_plot = st.selectbox("X-axis Factor:", avail_plot_feat_k, index=0, key="kmeans_plot_x_main") # Changed key
                                    with col_p2:
                                        y_opts_k = [f for f in avail_plot_feat_k if f != x_axis_k_plot]; y_idx_k = 0 if not y_opts_k else (avail_plot_feat_k.index(y_opts_k[0]) if y_opts_k[0] in avail_plot_feat_k else 0)
                                        if y_idx_k >= len(avail_plot_feat_k) or avail_plot_feat_k[y_idx_k] == x_axis_k_plot : y_idx_k = 1 if len(avail_plot_feat_k)>1 else 0
                                        y_axis_k_plot = st.selectbox("Y-axis Factor:", avail_plot_feat_k, index=y_idx_k, key="kmeans_plot_y_main") # Changed key
                                    if x_axis_k_plot and y_axis_k_plot and x_axis_k_plot != y_axis_k_plot:
                                         fig_cluster_k = plot_clusters(df_plot_k, [x_axis_k_plot, y_axis_k_plot], df_plot_k['Cluster'], st.session_state.model)
                                         if fig_cluster_k: st.plotly_chart(fig_cluster_k, use_container_width=True)
                                    elif x_axis_k_plot == y_axis_k_plot and len(avail_plot_feat_k) > 1: st.warning("Select two different factors for the cluster plot.")
                                    elif len(avail_plot_feat_k) < 2: st.warning("Need at least two numeric factors used in K-Means for a 2D plot.")
                               else: st.warning("Need at least 2 numeric factors used in K-Means for a 2D plot.")
                          else: st.warning(f"Data length vs cluster labels mismatch.")
                     else: st.warning("Cluster labels or features used for K-Means are unavailable for plotting.")
            except Exception as e: st.error(f"Error generating visualizations: {e}"); st.exception(e)

            st.write("**Claim Spoils (Downloads):**"); col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                 if st.session_state.model:
                      model_fn_dl = f"{st.session_state.selected_theme_name.lower().replace(' ', '_')}_{st.session_state.selected_model_type.replace(' ', '_').lower()}_strategy.joblib"
                      model_bdl_dl = {'model': st.session_state.model}
                      if st.session_state.scaler: model_bdl_dl['scaler'] = st.session_state.scaler
                      if st.session_state.model_feature_names: model_bdl_dl['feature_names'] = st.session_state.model_feature_names
                      download_file(model_bdl_dl, model_fn_dl, "Strategy Bundle", file_format='joblib')
                 else: st.info("No strategy to download.")
            with col_dl2:
                 res_df_dl = None; res_fn_dl = "outcome_data.csv"; res_lbl_dl = "Outcome Data"
                 if st.session_state.predictions is not None and st.session_state.selected_model_type != "K-Means Clustering" and st.session_state.X_test is not None:
                     res_df_dl = st.session_state.X_test.copy(); res_df_dl['Actual_Target'] = st.session_state.y_test.values; res_df_dl['Predicted_Target'] = st.session_state.predictions
                     if st.session_state.selected_model_type == "Logistic Regression" and st.session_state.pred_proba is not None:
                          if hasattr(st.session_state.model, 'classes_'):
                               for i, cl_lbl_dl in enumerate(st.session_state.model.classes_): res_df_dl[f'Probability_{cl_lbl_dl}'] = st.session_state.pred_proba[:, i]
                          else:
                               for i in range(st.session_state.pred_proba.shape[1]): res_df_dl[f'Probability_Class_{i}'] = st.session_state.pred_proba[:, i]
                     res_fn_dl = "test_predictions_output.csv"; res_lbl_dl = "Test Predictions"
                 elif 'cluster_labels' in st.session_state and st.session_state.cluster_labels is not None and st.session_state.selected_model_type == "K-Means Clustering" and st.session_state.preprocessed_data is not None:
                      res_df_dl = st.session_state.preprocessed_data.copy()
                      if len(res_df_dl) == len(st.session_state.cluster_labels):
                           res_df_dl['Cluster'] = st.session_state.cluster_labels; res_fn_dl = "clustered_data_output.csv"; res_lbl_dl = "Clustered Data"
                      else: st.warning("Cluster label alignment error for download."); res_df_dl = None
                 if res_df_dl is not None: download_file(res_df_dl, res_fn_dl, res_lbl_dl, file_format='csv')
                 else: st.info("No results data available to download.")
            st.success(theme["quotes"]["complete"])
            st.image(theme["gifs"]["footer"], caption=theme["quotes"]["quote"])
        elif st.session_state.model is not None: st.info("Assess the outcome (Step 6) first.")
        else: st.info("Execute a strategy (Step 5) and assess the outcome (Step 6) first.")

    # --- Footer ---
    st.markdown("---")
    st.markdown(f"<p style='text-align: center; color: {theme['colors']['accent']};'><i>{theme['quotes']['quote']}</i></p>", unsafe_allow_html=True)
