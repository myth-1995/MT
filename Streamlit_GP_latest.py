import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import linecache
import pickle

file_path = "data/scalar_gasp_Xend.pkl"
elevation_path = "data/synthetic_topo.asc"

# Open the file in binary read mode
with open(file_path, 'rb') as file:
    # Load the object from the file
    ScalarGaSP_Xend = pickle.load(file)

# Load elevation data

header = [linecache.getline(elevation_path, i) for i in range(1, 6)]
header_values = [float(h.split()[-1].strip()) for h in header]
ncols, nrows, xll, yll, cellsize = header_values
ncols = int(ncols)
nrows = int(nrows)

x_values = np.arange(xll, xll + (cellsize * ncols), cellsize)
y_values = np.arange(yll, yll + (cellsize * nrows), cellsize)
z_values = np.loadtxt(elevation_path, skiprows=6)
z_values = np.rot90(np.transpose(z_values))

# Initial parameters
x0 = 200
y0 = 1000
z0 = z_values[0, int(x0 / cellsize)]
mu_init = 0.23
xi_init = 1000

# Function to calculate final position based on sliders
def calculate_final_position(mu_slider, xi_slider):
    testing_input = np.ones((1, 2))
    testing_input[0][0] = mu_slider
    testing_input[0][1] = xi_slider
    x_fin = ScalarGaSP_Xend.predict(testing_input)[:, 0]
    index = int(x_fin[0] / cellsize)
    index = max(0, min(index, z_values.shape[1] - 1))
    z_fin = z_values[0, index]

    return x_fin, z_fin

# Define layout
st.title("Mass Point Model Visualization")
st.sidebar.title("Parameters")
mu_slider = st.sidebar.slider("Coulomb Friction", min_value=0.0, max_value=1.0, step=0.01, value=mu_init)
xi_slider = st.sidebar.slider("Turbulent Friction", min_value=0, max_value=2000, step=100, value=xi_init)



# Calculate final position
x_fin, z_fin = calculate_final_position(mu_slider, xi_slider)

# Plotting
fig, ax = plt.subplots(figsize=(10, 7))

np.random.seed(5)

# Add variation to elevation
variation = np.random.normal(loc=0, scale=mu_slider * 40, size=z_values.shape)
z_values_with_variation = z_values + variation

# Plot elevation profile
ax.plot(x_values, z_values_with_variation[0, :], color='black')

# Fill the area below the line plot with earthy brown color
ax.fill_between(x_values, z_values_with_variation[0, :], yll, color='#8B4513')

# Load the cropped hut image
hut_img = Image.open('cropped_hut_image.png')
hut_img = np.array(hut_img)

# Place the hut image at the bottom of the line plot
imagebox_hut = OffsetImage(hut_img, zoom=0.1)
ab_hut = AnnotationBbox(imagebox_hut, (3650, 115), frameon=False)
ax.add_artist(ab_hut)

offset_y = 65
# Highlight the final position with a brown dot
ax.scatter([x_fin[0]], z_fin + offset_y + (z_fin / 200), s=1000, c='brown', marker='o')  # Adjust the offset as needed

ax.set_xlabel('x')
ax.set_ylabel('z')
ax.set_xlim(0, 5000)  # Set fixed x-axis limits
ax.set_ylim(-25, 1400)  # Set fixed y-axis limits, adjust based on your data range
ax.set_title('Cross Section at Any y')

st.pyplot(fig)
