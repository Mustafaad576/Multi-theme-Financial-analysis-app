--- START OF FILE newer pf proj code.py ---

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
    "WWE Memes": {
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
            "welcome": "https://media.giphy.com/media/xT0BKiaM2VGJ4115Nu/giphy.gif",
            "footer": "https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif",
            "sidebar": "https://media.giphy.com/media/13V60d9GqTQv9a/giphy.gif"
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
            "welcome": "https://media.giphy.com/media/3o6ZtaO9BZHcOjmErm/giphy.gif",
            "footer": "https://media.giphy.com/media/jUwpNzg9IcyrK/giphy.gif",
            "sidebar": "https://media.giphy.com/media/3oEjHCWdU7F4sVOxPG/giphy.gif"
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
            "welcome": "https://media.giphy.com/media/l3vR9O7dhwEu5WzJK/giphy.gif",
            "footer": "https://media.giphy.com/media/3og0INyCmHlNylks9O/giphy.gif",
            "sidebar": "https://media.giphy.com/media/YTbZzCkRQCEJa/giphy.gif"
        },
        "icon": "🧪"
    }
}

# --- Initialize session state for selected_theme_name if not exists ---
if 'selected_theme_name' not in st.session_state:
    st.session_state.selected_theme_name = list(theme_options.keys())[0] # Default to first theme

# --- Get current theme from session state ---
theme = theme_options[st.session_state.selected_theme_name]

# --- Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title=f"{st.session_state.selected_theme_name} Financial Analysis",
    page_icon=theme["icon"],
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme Selection in Sidebar ---
with st.sidebar:
    st.image(theme["gifs"]["sidebar"], use_container_width=True)
    st.markdown(f"<h2 style='color:{theme['colors']['header']}; text-align: center;'>{st.session_state.selected_theme_name} Console</h2>", unsafe_allow_html=True)
    st.divider()

    new_selected_theme_name = st.selectbox(
        "Choose Theme:",
        list(theme_options.keys()),
        index=list(theme_options.keys()).index(st.session_state.selected_theme_name),
        key="theme_selector_sidebar"
    )
    if new_selected_theme_name != st.session_state.selected_theme_name:
        st.session_state.selected_theme_name = new_selected_theme_name
        st.rerun() # Rerun to apply new theme from top

# --- Apply Selected Theme CSS ---
GOOGLE_FONT = theme["font"]
FONT_URL = f"https://fonts.googleapis.com/css2?family={GOOGLE_FONT.replace(' ', '+')}:wght@400;700&display=swap"

# Define colors from theme for convenience (optional, but makes CSS block cleaner)
PRIMARY_COLOR = theme["colors"]["primary"]
ACCENT_COLOR = theme["colors"]["accent"]
BG_COLOR = theme["colors"]["bg"]
TEXT_COLOR = theme["colors"]["text"]
HEADER_COLOR = theme["colors"]["header"]
SIDEBAR_COLOR = theme["colors"]["sidebar"]
BORDER_COLOR = theme["colors"]["border"]
INPUT_BG_COLOR = theme["colors"].get("input_bg", "#303030") # Fallback for main area inputs
SIDEBAR_INPUT_BG_COLOR = theme["colors"].get("sidebar_input_bg", "#383838") # Fallback for sidebar inputs
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

    /* Main background and text */
    .stApp {{
        background-color: {BG_COLOR};
        color: {TEXT_COLOR};
    }}

    /* Sidebar */
    [data-testid="stSidebar"] > div:first-child {{
        background-color: {SIDEBAR_COLOR};
        border-right: 1px solid {BORDER_COLOR};
    }}
    /* Sidebar Headers & Text */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stText,
    [data-testid="stSidebar"] .stRadio label span {{
        color: {HEADER_COLOR} !important; /* Brighter text for sidebar */
        font-weight: 700; /* Bolder */
    }}
    /* Sidebar input widgets */
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stNumberInput input {{
        background-color: {SIDEBAR_INPUT_BG_COLOR} !important;
        color: {TEXT_COLOR} !important;
        border: 1px solid {BORDER_COLOR} !important;
    }}
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {{
        background-color: {SIDEBAR_INPUT_BG_COLOR} !important;
        color: {TEXT_COLOR} !important; /* Text color for selected item */
        border: 1px solid {BORDER_COLOR} !important;
        padding: 0.5em 0.6em !important;
        height: auto !important;
        min-height: 40px !important;
        line-height: 1.6 !important;
    }}
    [data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div > div {{ /* Inner div for text */
         color: {TEXT_COLOR} !important;
         overflow: visible !important;
         white-space: normal !important;
         height: auto !important;
    }}
    /* Styling for dropdown list items in sidebar */
    div[data-baseweb="popover"] ul li {{
        background-color: {SIDEBAR_INPUT_BG_COLOR} !important;
        color: {TEXT_COLOR} !important;
    }}
    div[data-baseweb="popover"] ul li:hover {{
        background-color: {PRIMARY_COLOR} !important;
        color: {ACCENT_COLOR} !important;
    }}

    [data-testid="stSidebar"] .stFileUploader label span {{
         color: {TEXT_COLOR} !important;
    }}

    /* Buttons */
    .stButton>button {{
        background-color: {PRIMARY_COLOR};
        color: {BUTTON_TEXT_COLOR}; 
        border: 1px solid {ACCENT_COLOR};
        border-radius: 5px;
        padding: 10px 22px;
        transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
        font-weight: 700;
        font-size: 1.05rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .stButton>button:hover {{
        background-color: {BUTTON_HOVER_BG};
        color: {BUTTON_HOVER_TEXT};
        border-color: {PRIMARY_COLOR if BUTTON_HOVER_BG == ACCENT_COLOR else ACCENT_COLOR};
    }}
     .stButton>button:focus {{
         box-shadow: 0 0 0 3px {ACCENT_COLOR}66; /* Accent color with alpha */
         outline: none;
    }}

    /* Titles and Headers in Main Area */
    h1, h2, h3 {{
        color: {HEADER_COLOR}; 
        font-weight: 700;
        text-shadow: 1px 1px 2px #00000033; /* Subtle shadow for dark text on light bg might be too much */
    }}
    h1 {{ /* Main title */
        border-bottom: 2px solid {PRIMARY_COLOR};
        padding-bottom: 0.6rem;
        text-align: center; 
    }}
     h3 {{ /* Step headers */
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-top: 1px solid {BORDER_COLOR};
        padding-top: 1rem;
        color: {ACCENT_COLOR}; /* Accent for step headers */
    }}

    /* Dataframe styling */
     .stDataFrame {{
        border: 1px solid {BORDER_COLOR};
        border-radius: 0px;
        background-color: {BG_COLOR}; 
    }}
     .stDataFrame thead th {{
        background-color: {PRIMARY_COLOR};
        color: {ACCENT_COLOR};
        font-weight: 700;
        text-transform: uppercase;
    }}
    .stDataFrame tbody td {{
        color: {TEXT_COLOR};
        background-color: {BG_COLOR}; /* Ensure cell background matches app background */
    }}
    .stDataFrame tbody tr:nth-child(even) td {{ /* Zebra striping for readability */
        background-color: color-mix(in srgb, {BG_COLOR} 90%, {TEXT_COLOR} 10%);
    }}


    /* Style metric labels/values */
    [data-testid="stMetricLabel"] {{
        color: color-mix(in srgb, {TEXT_COLOR} 70%, {BG_COLOR} 30%); /* Lighter gray for metric labels */
        font-size: 0.95rem;
        text-transform: uppercase;
    }}
    [data-testid="stMetricValue"] {{
        color: {ACCENT_COLOR}; 
        font-size: 1.3rem;
        font-weight: 700;
    }}

    /* Input widgets styling in main area */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div,
    .stMultiSelect>div>div {{
        border: 1px solid {BORDER_COLOR} !important;
        border-radius: 0px; 
        background-color: {INPUT_BG_COLOR} !important;
        color: {TEXT_COLOR} !important;
    }}
    .stSelectbox>div>div, .stMultiSelect>div>div {{ /* Adjust padding/appearance for select/multiselect */
         padding-top: 0.4rem;
         padding-bottom: 0.4rem;
    }}
    /* Slider track/thumb */
    .stSlider [data-baseweb="slider"] > div:nth-child(2) > div {{ /* Track */
         background: {PRIMARY_COLOR};
    }}
     .stSlider [data-baseweb="slider"] > div:nth-child(3) {{ /* Thumb */
         background: {ACCENT_COLOR};
         border: 2px solid {PRIMARY_COLOR};
    }}

    /* Style info/success/warning/error boxes */
    .stAlert {{
         border-radius: 0px; 
         border: 1px solid {ACCENT_COLOR};
         background-color: color-mix(in srgb, {BG_COLOR} 70%, #000000 30%); /* Dark transparent background */
    }}
    [data-testid="stAlert"] p {{
         color: {TEXT_COLOR};
    }}
    [data-testid="stAlert"][kind="info"] {{ border-left: 5px solid #4a90e2; }} 
    [data-testid="stAlert"][kind="success"] {{ border-left: 5px solid #50e3c2; }} 
    [data-testid="stAlert"][kind="warning"] {{ border-left: 5px solid {ACCENT_COLOR}; }} 
    [data-testid="stAlert"][kind="error"] {{ border-left: 5px solid {PRIMARY_COLOR}; }} 

    /* Center welcome screen elements */
    .welcome-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
</style>
""", unsafe_allow_html=True)


# --- Helper Functions ---#

def download_file(data, filename, label_key, file_format='csv'): # label_key for theme["quotes"]
    """Generates a download button for dataframes or models."""
    global theme # Access the global theme dictionary
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
            label=download_button_label,
            data=buffer,
            file_name=filename,
            mime=mime,
            key=f"download_{filename.replace('.', '_')}_{label_key.replace(' ', '_')}" # More unique key
        )
    except Exception as e: st.error(f"Error preparing {label_key} for download: {e}")


def plot_feature_importance(model, feature_names):
    """Plots feature importance for linear models."""
    global theme # Access the global theme dictionary
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
                 title=f"Feature Importance: What Influences the Outcome?", # Generic title
                 color_discrete_sequence=[theme["colors"]["accent"]],
                 labels={'importance': 'Influence (Coefficient/Value)', 'feature': 'Factor'})
    fig.update_layout(
        yaxis={'categoryorder':'total ascending'},
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=theme["colors"]["text"],
        title_font_color=theme["colors"]["header"],
        yaxis_tickfont_color=theme["colors"]["text"],
        xaxis_tickfont_color=theme["colors"]["text"],
        xaxis_title_font_color=theme["colors"]["text"],
        yaxis_title_font_color=theme["colors"]["text"],
        legend_font_color=theme["colors"]["text"]
    )
    fig.update_xaxes(gridcolor=theme["colors"]["border"])
    fig.update_yaxes(gridcolor=theme["colors"]["border"])
    return fig


def plot_clusters(df, features, cluster_labels, kmeans_model):
    """Plots K-Means clusters using the first two selected features."""
    global theme # Access the global theme dictionary
    if len(features) < 2: st.warning("Need at least two features selected to plot clusters."); return None

    cluster_labels_str = cluster_labels.astype(str)
    
    # Use a color sequence that contrasts well with the theme
    # For dark themes, brighter/pastel colors. For light themes, more saturated.
    # This is a simple heuristic, could be refined or made part of theme_options
    bg_lum = int(theme["colors"]["bg"][1:3], 16) * 0.299 + \
             int(theme["colors"]["bg"][3:5], 16) * 0.587 + \
             int(theme["colors"]["bg"][5:7], 16) * 0.114 # Calculate luminance
    
    if bg_lum < 128: # Dark background
        color_sequence = px.colors.qualitative.Pastel
    else: # Light background
        color_sequence = px.colors.qualitative.Bold

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
    # else: st.warning("Cannot plot centroids: Scaled feature names or centroids missing for plotting.") # Can be noisy

    fig.update_layout(
        legend_title_text='Group (Cluster)',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color=theme["colors"]["text"],
        title_font_color=theme["colors"]["header"],
        xaxis_tickfont_color=theme["colors"]["text"],
        yaxis_tickfont_color=theme["colors"]["text"],
        xaxis_title_font_color=theme["colors"]["text"],
        yaxis_title_font_color=theme["colors"]["text"],
        legend_font_color=theme["colors"]["text"],
        xaxis_gridcolor=theme["colors"]["border"],
        yaxis_gridcolor=theme["colors"]["border"]
        )
    return fig


# --- Initialize Session State ---
def init_session_state():
    """Initializes all required session state variables if they don't exist."""
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
        keys_to_reset = list(dict.fromkeys(keys_to_reset)) # Remove duplicates
    except ValueError: st.warning(f"Invalid reset level: {level}"); keys_to_reset = []

    reset_count = 0
    for key in keys_to_reset:
        if key in st.session_state and st.session_state[key] is not None:
            st.session_state[key] = None; reset_count += 1
    # if reset_count > 0: st.toast(f"Reset state from {level}.", icon="🧹")


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
        if st.button("Enter the Console", key="enter_app"):
            st.session_state.app_started = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# --- Main Application Logic ---
elif st.session_state.app_started:

    # --- Sidebar (Content defined after theme selection, but displayed by Streamlit) ---
    with st.sidebar:
        # Theme selector is already at the top of the sidebar logic
        st.divider() # After theme selector and title/gif

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
            ticker = st.text_input("Market Ticker(s) (e.g., AAPL, MSFT)", "AAPL", key="yahoo_ticker_input")
            col1, col2 = st.columns(2)
            with col1: period = st.selectbox("Period:", ['1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','ytd','max'], index=5, key="yahoo_period_select")
            with col2: interval = st.selectbox("Interval:", ['1d','5d','1wk','1mo','3mo'], index=0, key="yahoo_interval_select")

            if st.button("Scry the Market", key="fetch_yahoo_btn", use_container_width=True):
                if ticker:
                    if not all(c.isalnum() or c in ['-', '.', '^'] for c in ticker.replace(' ', '')): st.warning("Invalid ticker format.")
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
                                # Handle MultiIndex columns if multiple tickers are fetched
                                if isinstance(data_df.columns, pd.MultiIndex):
                                    if len(tickers_list) == 1: # Single ticker might still return MultiIndex if e.g. 'Adj Close' is different
                                        data_df.columns = data_df.columns.droplevel(1)
                                    else: # Multiple tickers, flatten columns
                                        data_df.columns = ["_".join(map(str, col)).strip().replace(' ', '_').replace('<', '').replace('>', '') for col in data_df.columns.values]
                                
                                # Ensure Datetime index is a column
                                if isinstance(data_df.index, pd.DatetimeIndex):
                                    data_df = data_df.reset_index()

                                # Convert potential date column to datetime if not already
                                date_col_to_convert = next((col for col in data_df.columns if col.lower() in ['date', 'datetime']), None)
                                if date_col_to_convert and not pd.api.types.is_datetime64_any_dtype(data_df[date_col_to_convert]):
                                     try: data_df[date_col_to_convert] = pd.to_datetime(data_df[date_col_to_convert])
                                     except Exception as date_e: st.warning(f"Could not parse date column '{date_col_to_convert}': {date_e}")
                                
                                # Drop rows where all price data (Open, High, Low, Close) might be NaN (common for multiple tickers)
                                price_cols = [col for col in data_df.columns if any(c.lower() in col.lower() for c in ['open', 'high', 'low', 'close'])]
                                if price_cols:
                                    data_df = data_df.dropna(subset=price_cols, how='all')


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
            
            # Find current index based on actual model type
            current_model_display_name = st.session_state.selected_model_type # This stores the actual model type
            if st.session_state.selected_model_type:
                try:
                    current_display_name_for_select = [k for k, v in model_options_map.items() if v == st.session_state.selected_model_type][0]
                    current_model_index = model_display_names.index(current_display_name_for_select)
                except (IndexError, ValueError):
                    current_model_index = 0 # Default if not found
            else:
                current_model_index = 0


            selected_display_name = st.selectbox(
                "Select Analytical Strategy:", model_display_names, index=current_model_index, key="model_select_display",
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
                    with st.spinner(f"Refining data... {theme['quotes']['quote'][:20]}..."): # Short quote
                        data_to_process = st.session_state.data.copy()
                        st.write("**Missing Values Before Refining:**")
                        missing_vals = data_to_process.isnull().sum(); missing_vals_df = missing_vals[missing_vals > 0].reset_index(name='count').rename(columns={'index':'Column'})
                        if not missing_vals_df.empty: st.dataframe(missing_vals_df)
                        else: st.write("No missing values found.")
                        
                        numeric_cols = data_to_process.select_dtypes(include=np.number).columns
                        categorical_cols = data_to_process.select_dtypes(exclude=np.number).columns
                        
                        imputed_numeric_cols = []; imputed_categorical_cols = []; potential_datetime_cols = []

                        # Impute Numeric
                        if not numeric_cols.empty:
                            cols_with_nan_numeric = numeric_cols[data_to_process[numeric_cols].isnull().any()].tolist()
                            if cols_with_nan_numeric:
                                num_imputer = SimpleImputer(strategy='mean')
                                data_to_process[cols_with_nan_numeric] = num_imputer.fit_transform(data_to_process[cols_with_nan_numeric])
                                imputed_numeric_cols = cols_with_nan_numeric
                                if imputed_numeric_cols: st.write(f"Imputed numeric columns (mean): `{', '.join(imputed_numeric_cols)}`")
                        
                        # Identify potential datetime columns among categoricals before imputation
                        if not categorical_cols.empty:
                            for col in categorical_cols:
                                try:
                                    # Check if a significant portion can be parsed as datetime
                                    if pd.to_datetime(data_to_process[col], errors='coerce').notna().mean() > 0.5 and not pd.api.types.is_datetime64_any_dtype(data_to_process[col]):
                                        potential_datetime_cols.append(col)
                                    elif pd.api.types.is_datetime64_any_dtype(data_to_process[col]): # Already datetime
                                        potential_datetime_cols.append(col)
                                except Exception:
                                    pass # Not easily parsable as datetime

                            # Impute Categorical (excluding identified datetime candidates)
                            cols_to_impute_cat = categorical_cols.difference(potential_datetime_cols)
                            if not cols_to_impute_cat.empty:
                                cols_with_nan_cat = cols_to_impute_cat[data_to_process[cols_to_impute_cat].isnull().any()].tolist()
                                if cols_with_nan_cat:
                                    cat_imputer = SimpleImputer(strategy='most_frequent')
                                    data_to_process[cols_with_nan_cat] = cat_imputer.fit_transform(data_to_process[cols_with_nan_cat])
                                    imputed_categorical_cols = cols_with_nan_cat
                                    if imputed_categorical_cols: st.write(f"Imputed categorical columns (mode): `{', '.join(imputed_categorical_cols)}`")

                        st.write("**Missing Values After Refining:**")
                        missing_vals_after = data_to_process.isnull().sum(); missing_vals_after_df = missing_vals_after[missing_vals_after > 0].reset_index(name='count').rename(columns={'index':'Column'})
                        if not missing_vals_after_df.empty: st.dataframe(missing_vals_after_df)
                        else: st.success("No missing values remain.")
                        
                        st.write("**Encoding Factors:**")
                        encoders = {}; encoded_cols = []
                        # Encode non-numeric columns that are not potential datetimes
                        cols_to_encode = data_to_process.select_dtypes(include=['object', 'category']).columns.difference(potential_datetime_cols)
                        for col in cols_to_encode:
                             n_unique = data_to_process[col].nunique()
                             if n_unique > 1 and n_unique < 100: # Heuristic for cardinality
                                  try: 
                                      le = LabelEncoder(); data_to_process[col] = le.fit_transform(data_to_process[col].astype(str))
                                      encoders[col] = le; encoded_cols.append(col)
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
                potential_features = df_processed.columns.tolist()
                
                # Exclude actual datetime columns from feature/target selection (they should be engineered first if used)
                datetime_cols = df_processed.select_dtypes(include=['datetime', 'datetime64[ns]', 'timedelta']).columns.tolist()
                non_datetime_features = [col for col in potential_features if col not in datetime_cols]

                numeric_features = df_processed[non_datetime_features].select_dtypes(include=np.number).columns.tolist()
                
                st.write("**Select Input Factors (X)**")
                default_feature_selection = numeric_features if numeric_features else non_datetime_features[:-1] if len(non_datetime_features) > 1 else non_datetime_features
                selected_features = st.multiselect("Select one or more factors:", non_datetime_features, default=default_feature_selection, key="feature_select_ms", help="Inputs for the strategy.")
                
                target_column = None
                if st.session_state.selected_model_type != "K-Means Clustering":
                    st.write("**Select Target Outcome (y)**")
                    potential_targets = [col for col in non_datetime_features if col not in selected_features]
                    default_target_guess = None
                    numeric_potential_targets = [t for t in potential_targets if t in numeric_features]
                    
                    common_targets_keywords = ['close', 'adj close', 'volume', 'target', 'label', 'signal', 'profit', 'return', 'price']
                    if numeric_potential_targets:
                        for kw in common_targets_keywords:
                            matches = [pt for pt in numeric_potential_targets if kw in pt.lower()]
                            if matches: default_target_guess = matches[0]; break
                        if not default_target_guess: default_target_guess = numeric_potential_targets[0] # Last numeric as fallback
                    elif potential_targets: default_target_guess = potential_targets[-1] if potential_targets else None


                    target_index = potential_targets.index(default_target_guess) if default_target_guess and default_target_guess in potential_targets else 0
                    target_column = st.selectbox("Select the target outcome:", potential_targets, index=target_index, key="target_select_sb", help="Outcome to predict.")
                    
                    if st.session_state.selected_model_type == "Logistic Regression" and target_column:
                         target_series = df_processed[target_column]
                         n_unique = target_series.nunique()
                         is_numeric = pd.api.types.is_numeric_dtype(target_series)
                         if n_unique < 2 : st.warning(f"Target '{target_column}' has only {n_unique} unique value (binary classification needs 2).")
                         elif n_unique > 10 and is_numeric: st.warning(f"Target '{target_column}' has {n_unique} unique numeric values. Consider binning for classification or if this is a regression problem.")
                         elif not is_numeric: st.warning(f"Target '{target_column}' is not numeric. Logistic regression usually expects numeric classes (e.g., 0 and 1).")
                         elif n_unique == 2 and is_numeric: st.success(f"Binary numeric target '{target_column}' selected.")


                if st.button("Confirm Factors", key="confirm_features_btn", use_container_width=True):
                    valid = True
                    if not selected_features: st.warning("Select at least one input factor."); valid = False
                    if st.session_state.selected_model_type != "K-Means Clustering":
                        if not target_column: st.warning("Select a target outcome."); valid = False
                        elif target_column in selected_features: st.error("Target cannot also be an input factor."); valid = False
                    
                    if st.session_state.selected_model_type == "Logistic Regression" and target_column and valid:
                        target_series = df_processed[target_column]
                        # Relaxed check: as long as it can be reasonably used. User responsible for binarization if needed.
                        if not pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 20: 
                             st.error("Target for classification is non-numeric with many unique values. Please ensure it's suitable (e.g., binarized or few classes)."); valid = False
                        elif target_series.nunique() < 2:
                             st.error("Target for classification must have at least 2 unique values."); valid = False
                    
                    if valid:
                        st.session_state.features = selected_features
                        st.session_state.target = target_column
                        st.session_state.feature_names = selected_features # Store original feature names before any transformation
                        st.success("Factors confirmed."); reset_downstream_state('feature_selection'); st.rerun()
            else: 
                 st.success("Step 3: Factors of Influence already selected.")
                 st.write("**Input Factors (X):**", f"`{', '.join(st.session_state.features)}`")
                 if st.session_state.target: st.write("**Target Outcome (y):**", f"`{st.session_state.target}`")
        else: st.info("Refine the data (Step 2) first.")

        # --- Step 4: Train/Test Split or Scaling ---
        step4_title_key = "Prepare the Battlefield (Split & Scale)" if st.session_state.selected_model_type != 'K-Means Clustering' else "Standardize Measures (Scale)"
        st.header(f"Step 4: {step4_title_key}")
        if st.session_state.features:
            step4_done = (st.session_state.X_train is not None) if st.session_state.selected_model_type != "K-Means Clustering" else (st.session_state.X_scaled_kmeans is not None)
            if not step4_done:
                if st.session_state.selected_model_type != "K-Means Clustering":
                    st.info("Divide forces into training and testing groups. Standardize numeric factor measures.")
                    test_size = st.slider("Reserve for Testing (%):", 10, 50, 20, 5, key="test_size_slider", format="%d%%", help="Portion of data held back for final evaluation.") / 100.0
                    random_state = st.number_input("Strategy Seed (Random State):", value=42, min_value=0, key="random_state_split", help="Ensures reproducible division.")
                    if st.button("Divide and Standardize", key="split_scale_btn", use_container_width=True):
                        try:
                            with st.spinner("Dividing forces..."):
                                X_df = st.session_state.preprocessed_data[st.session_state.features]
                                y_series = st.session_state.preprocessed_data[st.session_state.target]
                                
                                numeric_features_in_X = X_df.select_dtypes(include=np.number).columns.tolist()
                                X_processed = X_df.copy() # Start with a copy

                                if numeric_features_in_X:
                                    scaler = StandardScaler()
                                    X_processed[numeric_features_in_X] = scaler.fit_transform(X_df[numeric_features_in_X])
                                    st.session_state.scaler = scaler 
                                    st.write(f"Standardized numeric factors: `{', '.join(numeric_features_in_X)}`")
                                else:
                                    st.write("No numeric factors to standardize in X.")
                                    st.session_state.scaler = None # Ensure scaler is None if not used

                                st.session_state.model_feature_names = X_processed.columns.tolist() # Features used for model input after processing

                                stratify_target = y_series if st.session_state.selected_model_type == "Logistic Regression" and y_series.nunique() < 10 else None
                                X_train, X_test, y_train, y_test = train_test_split(X_processed, y_series, test_size=test_size, random_state=random_state, stratify=stratify_target)
                                
                                st.session_state.X_train, st.session_state.X_test, st.session_state.y_train, st.session_state.y_test = X_train, X_test, y_train, y_test
                                st.success("Forces divided and standardized.")
                                st.write(f"Training Forces: X {X_train.shape}, y {y_train.shape}"); st.write(f"Testing Forces: X {X_test.shape}, y {y_test.shape}")
                                
                                labels = ['Training Forces', 'Testing Forces']; sizes = [len(X_train), len(X_test)]
                                fig_split = px.pie(values=sizes, names=labels, title='Division of Forces', color_discrete_sequence=[theme["colors"]["primary"], theme["colors"]["accent"]])
                                fig_split.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=theme["colors"]["text"], legend_font_color=theme["colors"]["text"])
                                st.plotly_chart(fig_split, use_container_width=True)
                                reset_downstream_state('split_scale'); st.rerun()
                        except Exception as e: st.error(f"Error dividing forces: {e}"); st.exception(e)
                else: # K-Means Scaling
                     st.info("Standardize numeric factor measures for clustering.")
                     if st.button("Standardize Measures", key="scale_kmeans_btn", use_container_width=True):
                         try:
                              with st.spinner("Standardizing..."):
                                   X_df_kmeans = st.session_state.preprocessed_data[st.session_state.features]
                                   numeric_features_for_kmeans = X_df_kmeans.select_dtypes(include=np.number).columns.tolist()
                                   
                                   if not numeric_features_for_kmeans:
                                       st.error("Clustering requires numeric factors. None found in selected features."); st.stop()

                                   scaler_kmeans = StandardScaler()
                                   X_scaled_numeric_kmeans = scaler_kmeans.fit_transform(X_df_kmeans[numeric_features_for_kmeans])
                                   
                                   st.session_state.X_scaled_kmeans = X_scaled_numeric_kmeans
                                   st.session_state.scaler = scaler_kmeans # Store scaler for potential inverse transform or consistency
                                   st.session_state.scaled_feature_names_kmeans = numeric_features_for_kmeans # Store names of scaled features for plotting
                                   st.session_state.model_feature_names = numeric_features_for_kmeans # Kmeans uses these features

                                   st.success("Measures standardized."); st.write("Standardized Shape:", X_scaled_numeric_kmeans.shape); st.write(f"Factors standardized for K-Means: `{', '.join(numeric_features_for_kmeans)}`")
                                   reset_downstream_state('split_scale'); st.rerun()
                         except Exception as e: st.error(f"Error standardizing: {e}"); st.exception(e)
            else: 
                 st.success(f"Step 4: {step4_title_key} already completed.")
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
        prerequisites_met = (st.session_state.X_train is not None and st.session_state.y_train is not None) if st.session_state.selected_model_type != "K-Means Clustering" else (st.session_state.X_scaled_kmeans is not None)
        
        if prerequisites_met:
            if st.session_state.model is None:
                st.info(f"Configure parameters and execute the {st.session_state.selected_model_type} strategy.")
                model_params = {}
                with st.container(border=True):
                    st.write(f"**Configure {st.session_state.selected_model_type} Parameters:**")
                    if st.session_state.selected_model_type == "K-Means Clustering":
                        # Max K should be less than number of samples
                        max_k_val = st.session_state.X_scaled_kmeans.shape[0] -1 if st.session_state.X_scaled_kmeans.shape[0] > 1 else 1
                        max_k_val = min(15, max_k_val) # Cap at 15 for practicality
                        if max_k_val < 2: st.warning("Not enough samples for K-Means (K must be >= 2 and < n_samples)."); st.stop()

                        n_clusters = st.slider("Number of Groups (K):", min_value=2, max_value=max_k_val, value=min(3, max_k_val), key="kmeans_k_slider", help="Number of groups.")
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
                            with st.spinner(f"Executing strategy... {theme['quotes']['quote'][:25]}..."):
                                model_instance = None
                                # Ensure model_feature_names are from the X_train/X_scaled_kmeans columns
                                current_model_features = st.session_state.model_feature_names # Set in Step 4

                                if st.session_state.selected_model_type == "Linear Regression":
                                    model_instance = LinearRegression()
                                    # X_train already contains only the selected & processed features
                                    model_instance.fit(st.session_state.X_train, st.session_state.y_train)
                                elif st.session_state.selected_model_type == "Logistic Regression":
                                    if st.session_state.y_train.nunique() < 2:
                                        st.error("Classification target must have at least 2 unique values in training data."); st.stop()
                                    model_instance = LogisticRegression(random_state=42, solver='liblinear', **model_params)
                                    model_instance.fit(st.session_state.X_train, st.session_state.y_train)
                                elif st.session_state.selected_model_type == "K-Means Clustering":
                                    model_instance = KMeans(n_clusters=model_params['n_clusters'], random_state=42, n_init=10) # Use explicit n_clusters
                                    model_instance.fit(st.session_state.X_scaled_kmeans)
                                
                                st.session_state.model = model_instance
                                # st.session_state.model_feature_names was already set in step 4 with processed feature names.
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
            elif not ((st.session_state.X_train is not None and st.session_state.y_train is not None) if st.session_state.selected_model_type != "K-Means Clustering" else (st.session_state.X_scaled_kmeans is not None)):
                 st.info(f"{step4_title_key} (Step 4) first.")
            else: st.info("Complete prior steps.")


        # --- Step 6: Evaluation ---
        st.header("Step 6: Assess the Outcome")
        if st.session_state.model is not None:
             if st.session_state.metrics is None:
                st.info(f"Assess the outcome of the {st.session_state.selected_model_type} strategy.")
                if st.button("Assess Outcome", key="evaluate_model_btn", use_container_width=True):
                    try:
                        with st.spinner("Assessing outcome..."):
                            metrics_dict = {}; current_predictions = None; current_pred_proba = None
                            
                            # Ensure model_feature_names are available and consistent with X_test
                            if 'model_feature_names' not in st.session_state or not st.session_state.model_feature_names:
                                st.error("Model feature names not found. Re-run previous steps if necessary."); st.stop()
                            
                            model_features_used_for_training = st.session_state.model_feature_names

                            if st.session_state.selected_model_type != "K-Means Clustering":
                                 if st.session_state.X_test is None or st.session_state.y_test is None: 
                                     st.error("Testing forces not found. Run Step 4."); st.stop()
                                 
                                 # Ensure X_test has the same columns as X_train (which model was trained on)
                                 try:
                                     X_test_eval = st.session_state.X_test[model_features_used_for_training]
                                 except KeyError as e:
                                     st.error(f"Feature mismatch during evaluation. Model trained on {model_features_used_for_training}, "
                                              f"but X_test has {st.session_state.X_test.columns.tolist()}. Error: {e}"); st.stop()
                            
                            st.write("**Outcome Assessment:**")
                            if st.session_state.selected_model_type == "Linear Regression":
                                current_predictions = st.session_state.model.predict(X_test_eval)
                                metrics_dict['MSE'] = mean_squared_error(st.session_state.y_test, current_predictions)
                                metrics_dict['MAE'] = mean_absolute_error(st.session_state.y_test, current_predictions)
                                metrics_dict['R2 Score'] = r2_score(st.session_state.y_test, current_predictions)
                                col1, col2, col3 = st.columns(3)
                                col1.metric("MSE", f"{metrics_dict['MSE']:.4f}", help="Mean Squared Error"); 
                                col2.metric("MAE", f"{metrics_dict['MAE']:.4f}", help="Mean Absolute Error"); 
                                col3.metric("R2 Score", f"{metrics_dict['R2 Score']:.4f}", help="R-squared")
                            elif st.session_state.selected_model_type == "Logistic Regression":
                                current_predictions = st.session_state.model.predict(X_test_eval)
                                try: current_pred_proba = st.session_state.model.predict_proba(X_test_eval)
                                except AttributeError: current_pred_proba = None
                                metrics_dict['Accuracy'] = accuracy_score(st.session_state.y_test, current_predictions)
                                cm = confusion_matrix(st.session_state.y_test, current_predictions)
                                metrics_dict['Confusion Matrix'] = cm.tolist()
                                st.metric("Accuracy", f"{metrics_dict['Accuracy']:.4f}", help="Prediction Accuracy")
                                fig_cm = px.imshow(cm, text_auto=True, aspect="auto", labels=dict(x="Predicted", y="True", color="Count"), 
                                                   x=[str(c) for c in st.session_state.model.classes_], 
                                                   y=[str(c) for c in st.session_state.model.classes_], 
                                                   title="Confusion Matrix", 
                                                   color_continuous_scale=[[0, theme["colors"]["bg"]], [1, theme["colors"]["primary"]]])
                                fig_cm.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', 
                                                     font_color=theme["colors"]["text"], title_font_color=theme["colors"]["header"],
                                                     xaxis_tickfont_color=theme["colors"]["text"], yaxis_tickfont_color=theme["colors"]["text"])
                                st.plotly_chart(fig_cm, use_container_width=True)
                            elif st.session_state.selected_model_type == "K-Means Clustering":
                                labels = st.session_state.model.labels_
                                st.session_state.cluster_labels = labels # Save for plotting
                                n_labels = len(np.unique(labels))
                                n_samples = st.session_state.X_scaled_kmeans.shape[0]
                                if n_labels > 1 and n_labels < n_samples : # Silhouette score requires 1 < n_labels < n_samples
                                    metrics_dict['Silhouette Score'] = silhouette_score(st.session_state.X_scaled_kmeans, labels)
                                else: 
                                    metrics_dict['Silhouette Score'] = None
                                    st.warning(f"Silhouette Score N/A (requires 1 < K < n_samples; K={n_labels}, n_samples={n_samples}).")
                                metrics_dict['Inertia (WCSS)'] = st.session_state.model.inertia_
                                col1, col2 = st.columns(2)
                                ss_val = metrics_dict.get('Silhouette Score')
                                col1.metric("Silhouette Score", f"{ss_val:.4f}" if ss_val is not None else "N/A", help="Cluster separation (-1 to 1)")
                                col2.metric("Inertia (WCSS)", f"{metrics_dict['Inertia (WCSS)']:.2f}", help="Cluster compactness (Lower is better)")
                            
                            st.session_state.predictions = current_predictions
                            st.session_state.pred_proba = current_pred_proba
                            st.session_state.metrics = metrics_dict
                            st.success("Outcome assessed."); reset_downstream_state('evaluate'); st.rerun()
                    except Exception as e: st.error(f"Error assessing outcome: {e}"); st.exception(e)
             else: 
                 st.success("Step 6: Outcome already assessed.")
                 with st.expander("Show Outcome Assessment", expanded=True):
                     metrics_display = st.session_state.metrics
                     if st.session_state.selected_model_type == "Linear Regression":
                         col1, col2, col3 = st.columns(3)
                         col1.metric("MSE", f"{metrics_display.get('MSE', 'N/A'):.4f}" if isinstance(metrics_display.get('MSE'), (int, float)) else "N/A")
                         col2.metric("MAE", f"{metrics_display.get('MAE', 'N/A'):.4f}" if isinstance(metrics_display.get('MAE'), (int, float)) else "N/A")
                         col3.metric("R2 Score", f"{metrics_display.get('R2 Score', 'N/A'):.4f}" if isinstance(metrics_display.get('R2 Score'), (int, float)) else "N/A")
                     elif st.session_state.selected_model_type == "Logistic Regression":
                          st.metric("Accuracy", f"{metrics_display.get('Accuracy', 'N/A'):.4f}" if isinstance(metrics_display.get('Accuracy'), (int, float)) else "N/A")
                          cm_list = metrics_display.get('Confusion Matrix')
                          if cm_list and hasattr(st.session_state.model, 'classes_'):
                               cm_np = np.array(cm_list)
                               fig_cm_disp = px.imshow(cm_np, text_auto=True, aspect="auto", labels=dict(x="Predicted", y="True", color="Count"), 
                                                       x=[str(c) for c in st.session_state.model.classes_], y=[str(c) for c in st.session_state.model.classes_], 
                                                       title="Confusion Matrix", color_continuous_scale=[[0, theme["colors"]["bg"]], [1, theme["colors"]["primary"]]])
                               fig_cm_disp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=theme["colors"]["text"], title_font_color=theme["colors"]["header"], xaxis_tickfont_color=theme["colors"]["text"], yaxis_tickfont_color=theme["colors"]["text"])
                               st.plotly_chart(fig_cm_disp, use_container_width=True)
                     elif st.session_state.selected_model_type == "K-Means Clustering":
                          col1, col2 = st.columns(2)
                          ss_disp = metrics_display.get('Silhouette Score')
                          col1.metric("Silhouette Score", f"{ss_disp:.4f}" if ss_disp is not None else "N/A")
                          col2.metric("Inertia (WCSS)", f"{metrics_display.get('Inertia (WCSS)', 'N/A'):.2f}" if isinstance(metrics_display.get('Inertia (WCSS)'), (int, float)) else "N/A")
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
                             fig_imp = plot_feature_importance(st.session_state.model, st.session_state.model_feature_names)
                             if fig_imp: st.plotly_chart(fig_imp, use_container_width=True)
                         # else: st.info("Feature importance cannot be plotted (model type or features not suitable).") # Can be noisy
                         
                         if st.session_state.selected_model_type == "Linear Regression":
                             results_df_plot = pd.DataFrame({'Actual': st.session_state.y_test, 'Predicted': st.session_state.predictions})
                             fig_pred_plot = px.scatter(results_df_plot, x='Actual', y='Predicted', title='Actual vs. Predicted Values', 
                                                labels={'Actual': 'Actual Value', 'Predicted': 'Predicted Value'}, 
                                                trendline='ols', trendline_color_override=theme["colors"]["accent"], hover_data=results_df_plot.columns)
                             min_val = min(results_df_plot['Actual'].min(), results_df_plot['Predicted'].min()); max_val = max(results_df_plot['Actual'].max(), results_df_plot['Predicted'].max())
                             fig_pred_plot.add_shape(type='line', x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color=theme["colors"]["text"], dash='dash'))
                             fig_pred_plot.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color=theme["colors"]["text"], title_font_color=theme["colors"]["header"], xaxis_tickfont_color=theme["colors"]["text"], yaxis_tickfont_color=theme["colors"]["text"], xaxis_gridcolor=theme["colors"]["border"], yaxis_gridcolor=theme["colors"]["border"])
                             st.plotly_chart(fig_pred_plot, use_container_width=True)
                         elif st.session_state.selected_model_type == "Logistic Regression": st.info("Confusion Matrix shown during assessment (Step 6). Further visualizations can be added here.")
                     else: st.warning("Predictions/test data unavailable for visualization.")
                elif st.session_state.selected_model_type == "K-Means Clustering":
                     if 'cluster_labels' in st.session_state and st.session_state.cluster_labels is not None and 'scaled_feature_names_kmeans' in st.session_state and st.session_state.scaled_feature_names_kmeans:
                          # Plotting K-Means requires original unscaled data for interpretability, but with cluster labels.
                          # We use scaled_feature_names_kmeans to select from preprocessed_data for plotting.
                          df_plot_kmeans = st.session_state.preprocessed_data.copy() 
                          
                          if len(df_plot_kmeans) == len(st.session_state.cluster_labels):
                               df_plot_kmeans['Cluster'] = st.session_state.cluster_labels.astype(str)
                               
                               # Features for plotting should be from those used in K-Means (scaled_feature_names_kmeans)
                               available_plot_features = st.session_state.scaled_feature_names_kmeans
                               
                               if len(available_plot_features) >= 2:
                                    st.write("**Select factors for Cluster Plot (from features used in K-Means):**")
                                    col1_plot, col2_plot = st.columns(2)
                                    with col1_plot: x_axis_kmeans = st.selectbox("X-axis Factor:", available_plot_features, index=0, key="kmeans_plot_x_axis")
                                    with col2_plot: 
                                        y_options_kmeans = [f for f in available_plot_features if f != x_axis_kmeans]
                                        y_index_kmeans = 0 if not y_options_kmeans else (available_plot_features.index(y_options_kmeans[0]) if y_options_kmeans[0] in available_plot_features else 0)
                                        # Ensure y_index_kmeans is valid
                                        if y_index_kmeans >= len(available_plot_features) or available_plot_features[y_index_kmeans] == x_axis_kmeans : y_index_kmeans = 1 if len(available_plot_features)>1 else 0

                                        y_axis_kmeans = st.selectbox("Y-axis Factor:", available_plot_features, index=y_index_kmeans, key="kmeans_plot_y_axis")

                                    if x_axis_kmeans and y_axis_kmeans and x_axis_kmeans != y_axis_kmeans:
                                         # Pass df_plot_kmeans (original data + cluster labels) and selected features [x_axis_kmeans, y_axis_kmeans]
                                         # plot_clusters expects centroids to be plotted against the *scaled* features, so it internally uses scaled_feature_names_kmeans
                                         fig_cluster_plot = plot_clusters(df_plot_kmeans, [x_axis_kmeans, y_axis_kmeans], df_plot_kmeans['Cluster'], st.session_state.model)
                                         if fig_cluster_plot: st.plotly_chart(fig_cluster_plot, use_container_width=True)
                                    elif x_axis_kmeans == y_axis_kmeans and len(available_plot_features) > 1: st.warning("Select two different factors for the cluster plot.")
                                    elif len(available_plot_features) < 2: st.warning("Need at least two numeric factors used in K-Means for a 2D plot.")

                               else: st.warning("Need at least 2 numeric factors used in K-Means for a 2D plot.")
                          else: st.warning(f"Data length ({len(df_plot_kmeans)}) vs cluster labels ({len(st.session_state.cluster_labels)}) mismatch.")
                     else: st.warning("Cluster labels or features used for K-Means are unavailable for plotting.")
            except Exception as e: st.error(f"Error generating visualizations: {e}"); st.exception(e)

            st.write("**Claim Spoils (Downloads):**")
            col1_dl, col2_dl = st.columns(2)
            with col1_dl:
                 if st.session_state.model:
                      model_filename_dl = f"{st.session_state.selected_theme_name.lower().replace(' ', '_')}_{st.session_state.selected_model_type.replace(' ', '_').lower()}_strategy.joblib"
                      model_bundle_dl = {'model': st.session_state.model}
                      if st.session_state.scaler: model_bundle_dl['scaler'] = st.session_state.scaler
                      if st.session_state.model_feature_names: model_bundle_dl['feature_names'] = st.session_state.model_feature_names # These are the processed feature names
                      download_file(model_bundle_dl, model_filename_dl, "Strategy Bundle", file_format='joblib')
                 else: st.info("No strategy to download.")
            with col2_dl:
                 results_df_download_dl = None; results_filename_dl = "outcome_data.csv"; results_label_dl = "Outcome Data"
                 if st.session_state.predictions is not None and st.session_state.selected_model_type != "K-Means Clustering" and st.session_state.X_test is not None:
                     # Use original X_test features for better interpretability if possible, by inverting scaling
                     # For simplicity, we'll download the X_test as is (which is scaled/processed)
                     results_df_download_dl = st.session_state.X_test.copy()
                     results_df_download_dl['Actual_Target'] = st.session_state.y_test.values # y_test is original scale
                     results_df_download_dl['Predicted_Target'] = st.session_state.predictions
                     
                     if st.session_state.selected_model_type == "Logistic Regression" and st.session_state.pred_proba is not None:
                          if hasattr(st.session_state.model, 'classes_'):
                               for i, class_label_dl in enumerate(st.session_state.model.classes_): results_df_download_dl[f'Probability_{class_label_dl}'] = st.session_state.pred_proba[:, i]
                          else: # Fallback if classes_ attribute is not standard
                               for i in range(st.session_state.pred_proba.shape[1]): results_df_download_dl[f'Probability_Class_{i}'] = st.session_state.pred_proba[:, i]
                     results_filename_dl = "test_predictions_output.csv"; results_label_dl = "Test Predictions"

                 elif 'cluster_labels' in st.session_state and st.session_state.cluster_labels is not None and st.session_state.selected_model_type == "K-Means Clustering" and st.session_state.preprocessed_data is not None:
                      results_df_download_dl = st.session_state.preprocessed_data.copy() # Download original preprocessed data
                      if len(results_df_download_dl) == len(st.session_state.cluster_labels):
                           results_df_download_dl['Cluster'] = st.session_state.cluster_labels
                           results_filename_dl = "clustered_data_output.csv"; results_label_dl = "Clustered Data"
                      else: st.warning("Cluster label alignment error for download."); results_df_download_dl = None
                 
                 if results_df_download_dl is not None: download_file(results_df_download_dl, results_filename_dl, results_label_dl, file_format='csv')
                 else: st.info("No results data available to download.")

            st.success(theme["quotes"]["complete"])
            st.image(theme["gifs"]["footer"], caption=theme["quotes"]["quote"])

        elif st.session_state.model is not None: st.info("Assess the outcome (Step 6) first.")
        else: st.info("Execute a strategy (Step 5) and assess the outcome (Step 6) first.")

    # --- Footer ---
    st.markdown("---")
    st.markdown(f"<p style='text-align: center; color: {theme['colors']['accent']};'><i>{theme['quotes']['quote']}</i></p>", unsafe_allow_html=True)

--- END OF FILE newer pf proj code.py ---
