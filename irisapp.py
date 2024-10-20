import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Add custom CSS for background image
st.markdown("""
    <style>
    body {
        background-image: url'images.jpeg');
        background-size: cover;
    }
    .prediction-text {
        color: white;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Set the app title
st.title('Iris Flower Species Prediction')

# Define the input fields for the model
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, step=0.1)

# When the user clicks the predict button
if st.button('Predict'):
    # Prepare the input data
    data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    scaled_data = scaler.transform(data)

    # Make a prediction
    prediction = model.predict(scaled_data)
    species = ['Setosa', 'Versicolor', 'Virginica']

    # Display the result
    predicted_species = species[prediction[0]]
    st.markdown(f'<div class="prediction-text">The predicted species is: {predicted_species}</div>', unsafe_allow_html=True)

    # Display corresponding flower image based on prediction
    if predicted_species == 'Setosa':
        st.image('iris-setosa.png', caption='Setosa')
    elif predicted_species == 'Versicolor':
        st.image('iris-machinelearning.png', caption='Versicolor')
    else:
        st.image('iris-verginica.png', caption='Virginica')
