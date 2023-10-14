import streamlit as st
from fastai.vision.all import load_learner, PILImage
from PIL import Image

# Initialize Streamlit
st.title("Dog and Cat Breed Classifier")
st.write("This application will predict the breed of the dog or cat")
st.write("Upload a picture of a dog or cat")

uploaded_file = st.file_uploader("Choose a image file", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Load and use the FastAI model
    try:
        model = load_learner("model.pkl")  # Replace with your model path
        img = PILImage.create(uploaded_file)
        # `idx` contains the index of the predicted category
        prediction, idx, probs = model.predict(img)

        st.write(f"Prediction: {prediction}")
        # Use `idx` to index into `probs`, and convert tensor to Python scalar
        st.write(f"Probability: {probs[idx].item():.04f}")
        # Add a chart of the probabilities
        breed_names = model.dls.vocab

        # Convert tensor to Python list
        probs = [probs[i].item() for i in range(len(probs))]
        probs = dict(zip(breed_names, probs))

        st.bar_chart(probs)
    except Exception as e:
        st.write(f"Error: {e}")
