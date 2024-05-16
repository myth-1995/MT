import numpy as np
import streamlit as st
from psimpy.simulator import MassPointModel
import linecache
import pickle

file_path = "scalar_gasp_Xend.pkl"

# Open the file in binary read mode
with open(file_path, 'rb') as file:
    # Load the object from the file
    ScalarGaSP_Xend = pickle.load(file)

# Load elevation data
elevation_path = 'C:\\Users\\vmith\\Downloads\\Psimpy_trials\\Trial_I\\tests\\data\\synthetic_topo.asc'
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
    testing_input = np.ones((1,2))
    testing_input[0][0] = mu_slider
    testing_input[0][1] = xi_slider
    x_fin = ScalarGaSP_Xend.predict(testing_input)[:,0]
    #print(x_fin)
    index = int(x_fin[0] / cellsize)
    index = max(0, min(index, z_values.shape[1] - 1))
    z_fin = z_values[0, index]
    #print(z_fin)
    

   


    return x_fin, z_fin

# Define layout
st.title("Mass Point Model Visualization")
st.sidebar.title("Parameters")
mu_slider = st.sidebar.slider("Coulomb Friction", min_value=0.0, max_value=1.0, step=0.01, value=mu_init)
xi_slider = st.sidebar.slider("Turbulent Friction", min_value=0, max_value=2000, step=100, value=xi_init)

# Calculate final position
x_fin, z_fin = calculate_final_position(mu_slider, xi_slider)

# Plotting
st.plotly_chart({
    'data': [
        {'x': x_values, 'y': z_values[0, :], 'type': 'scatter', 'name': 'Data'},
        {'x': [x_fin[0]], 'y': [z_fin], 'mode': 'markers', 'marker': {'size': 10, 'color': 'red'}, 'name': 'Point of Interest'}
    ],
    'layout': {
        'xaxis': {'title': 'x'},
        'yaxis': {'title': 'z'},
        'title': 'Cross Section at Any y',
        'margin': {'l': 40, 'b': 40, 't': 40, 'r': 40},
        'legend': {'x': 0, 'y': 1},
        'hovermode': 'closest'
    }
})
