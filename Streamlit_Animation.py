import streamlit as st
import base64
import os

# Constants
gif_directory = "animations_score_mit_Improvements"
mu_values = [0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24, 0.32, 0.48]
xi_values = [100, 200, 400, 600, 800, 1200, 1600, 1800, 2000, 2200]

# Initial values
initial_mu = 0.48
initial_xi = 1200

@st.cache_data
def load_gif(filepath):
    try:
        with open(filepath, "rb") as f:
            gif_bytes = f.read()
        gif_base64 = base64.b64encode(gif_bytes).decode("utf-8")
        return gif_base64
    except Exception as e:
        return None

# Title for the animation
st.title("Idealized Landslide Simulation")

# Dropdown widgets for mu and xi
mu_dropdown = st.selectbox("Select Colomb friction :", options=mu_values, index=mu_values.index(initial_mu))
xi_dropdown = st.selectbox("Select Turbulent Friction :", options=xi_values, index=xi_values.index(initial_xi))

# Construct filename based on selected mu and xi values
filename = f'speed_up_30_animation_mu_{mu_dropdown}_xi_{xi_dropdown}.gif'
filepath = os.path.join(gif_directory, filename)

# Display the animation if it exists, otherwise show a message
gif_base64 = load_gif(filepath)

if gif_base64:
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center;">
            <img src="data:image/gif;base64,{gif_base64}" alt="Landslide Animation">
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.write("No animation found for the selected parameters.")
