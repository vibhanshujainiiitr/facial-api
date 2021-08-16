import streamlit as st
from PIL import Image
from deepface import DeepFace
import pandas as pd
import skimage.io
from run import *
import cv2

st.title('CKM VIGIL Face API')
st.subheader("Facial Landmark Detection")
st.write('CKM VIGIL Face API is solution that estimates 468 3D face landmarks in real-time. It only requires a simple face image.')
st.subheader("Please upload your image")

uploaded_file = st.file_uploader("Choose an Image 1", type=["jpg", "png", "jpeg"], key = "image1")
uploaded = st.file_uploader("Choose an Image 2", type=["jpg", "png", "jpeg"], key = 'image2')

options_selected = st.sidebar.multiselect('Which result you want to get', ["Keypoints", "Characteristics", "Emotion", "Age", "Gender", "Race", "Distance", "Angle"])

with st.sidebar:
    # st.write("How good results are?")
    # slider_val = st.slider("0: Worst - 100: Best")
    # checkbox_val = st.checkbox("Giving wrong results!")
    
    # # Every form must have a submit button.
    # submitted = st.form_submit_button("Submit")
    # if submitted:
    #     st.write("slider", slider_val, "checkbox", checkbox_val)
    #     st.write("Thanks for submitting the form")
    st.write("Check our more projects on [ckmvigil.in/project](https://ckmvigil.in/projects)")

    # image = Image.open("https://ckmvigil.in/assets/images/logo/logo_full.png")

    # st.image(image)

if uploaded_file is not None:
    st.image(uploaded_file, caption='Original Image')
if uploaded is not None:
    st.image(uploaded, caption='Original Image')

if uploaded_file is not None:
    # st.image(uploaded_file, caption='Original Image')
    distance_dict, angle_dict, clone, keypoints = main(uploaded_file)
    emotion_image = skimage.io.imread(uploaded_file, plugin='matplotlib')
    
    obj = DeepFace.analyze(img_path = emotion_image, actions = ['age', 'gender', 'race', 'emotion'])
    if "Keypoints" in options_selected:
        st.image(keypoints, caption='Image with Keypoints')
    if "Characteristics" in options_selected:
        st.image(clone, caption="Characteristics")
    if "Emotion" in options_selected:
        emotion_writeup = st.text("Processing Image...")
        emotion_writeup.subheader("Detected Emotion is {}".format(max(obj["emotion"], key = lambda x: obj["emotion"][x])))
    if "Age" in options_selected:
        age_writeup = st.text("Processing Image...")
        age_writeup.subheader("Detected age is {}".format(obj["age"]))
    if "Gender" in options_selected:
        gender_writeup = st.text("Processing Image...")
        gender_writeup.subheader("Detected gender is {}".format(obj["gender"]))
    if "Race" in options_selected:
        race_writeup = st.text("Processing Image...")
        race_writeup.subheader("Detected race is {}".format(max(obj["race"], key = lambda x: obj["race"][x]))) 
    if "Distance" in options_selected:
        st.subheader("The distance between key points")
        df = pd.DataFrame(distance_dict, index=["distance"])
        df.T
    if "Angle" in options_selected:
        st.subheader("The important angles")
        df = pd.DataFrame(angle_dict, index=["angle"])
        df.T


if uploaded is not None:
    distance_dict, angle_dict, clone, keypoints = main(uploaded)
    emotion_image = skimage.io.imread(uploaded, plugin='matplotlib')
    st.write(emotion_image.shape)
    obj = DeepFace.analyze(img_path = emotion_image, actions = ['age', 'gender', 'race', 'emotion'])
    if "Keypoints" in options_selected:
        st.image(keypoints, caption='Image with Keypoints')
    if "Characteristics" in options_selected:
        st.image(clone, caption="Characteristics")
    if "Emotion" in options_selected:
        emotion_writeup = st.text("Processing Image...")
        emotion_writeup.subheader("Detected Emotion is {}".format(max(obj["emotion"], key = lambda x: obj["emotion"][x])))
    if "Age" in options_selected:
        age_writeup = st.text("Processing Image...")
        age_writeup.subheader("Detected age is {}".format(obj["age"]))
    if "Gender" in options_selected:
        gender_writeup = st.text("Processing Image...")
        gender_writeup.subheader("Detected gender is {}".format(obj["gender"]))
    if "Race" in options_selected:
        race_writeup = st.text("Processing Image...")
        race_writeup.subheader("Detected race is {}".format(max(obj["race"], key = lambda x: obj["race"][x]))) 
    if "Distance" in options_selected:
        st.subheader("The distance between key points")
        df = pd.DataFrame(distance_dict, index=["distance"])
        df.T
    if "Angle" in options_selected:
        st.subheader("The important angles")
        df = pd.DataFrame(angle_dict, index=["angle"])
        df.T