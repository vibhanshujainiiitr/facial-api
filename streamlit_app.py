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
st.subheader("try it out")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
options_selected = st.sidebar.multiselect('Select What to show on image', ["keypoints", "characteristics", "emotion", "age", "gender", "race", "distance", "angle"])

with st.sidebar.form("my_form"):
    st.write("How good results are?")
    slider_val = st.slider("0 very bad to 100 very good")
    checkbox_val = st.checkbox("need improvement?")
    
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("slider", slider_val, "checkbox", checkbox_val)
        st.write("Thanks for submitting the form")

if uploaded_file is not None:
    st.image(uploaded_file, caption='original image')
    distance_dict, angle_dict, clone, keypoints = main(uploaded_file)
    emotion_image = skimage.io.imread(uploaded_file, plugin='matplotlib')
    obj = DeepFace.analyze(img_path = emotion_image, actions = ['age', 'gender', 'race', 'emotion'])
    if "keypoints" in options_selected:
        st.image(keypoints, caption='keypoints Image')
    if "characteristics" in options_selected:
        st.image(clone, caption="Characteristics")
    if "emotion" in options_selected:
        emotion_writeup = st.text("Processing Image...")
        emotion_writeup.text("Detected Emotion is {}".format(max(obj["emotion"], key = lambda x: obj["emotion"][x])))
    if "age" in options_selected:
        age_writeup = st.text("Processing Image...")
        age_writeup.text("Detected age is {}".format(obj["age"]))
    if "gender" in options_selected:
        gender_writeup = st.text("Processing Image...")
        gender_writeup.text("Detected gender is {}".format(obj["gender"]))
    if "race" in options_selected:
        race_writeup = st.text("Processing Image...")
        race_writeup.text("Detected race is {}".format(max(obj["race"], key = lambda x: obj["race"][x]))) 
    if "distance" in options_selected:
        df = pd.DataFrame(distance_dict, index=["distance"])
        df.T
    if "angle" in options_selected:
        df = pd.DataFrame(angle_dict, index=["angle"])
        df.T