import streamlit as st
import base64,numpy as np
import cv2 
from imageprocessing import equalize_hist,clahe,adjust_gamma,white_balance


col1, col2 = st.columns( [0.95, 0.05])
with col1:               # To display the header text using css style
    st.markdown(""" <style> .font {
    font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
    </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font">Upload your photo here...</p>', unsafe_allow_html=True)
    
st.sidebar.markdown('<p class="font">My First Photo Converter App</p>', unsafe_allow_html=True)
with st.sidebar.expander("About the App"):
    st.write("""
        Use this simple app to convert your favorite photo to a pencil sketch, a grayscale image or an image with blurring effect.  \n  \nThis app was created by My Data Talk as a side project to learn Streamlit and computer vision. Hope you enjoy!
     """)
uploaded_file = st.file_uploader("", type=['jpg','png','jpeg'])
if uploaded_file is not None:
    #image=cv2.imread(uploaded_file)
    
    image=uploaded_file.getvalue()

    jpg_as_np = np.frombuffer(image, dtype=np.uint8)
    image = cv2.imdecode(jpg_as_np, flags=1)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(type(image))

    
    col1, col2 = st.columns( [0.5, 0.5])
    with col1:
        st.markdown('<p style="text-align: center;">Before</p>',unsafe_allow_html=True)
        st.image(image,width=400)  

    with col2:
        st.markdown('<p style="text-align: center;">After</p>',unsafe_allow_html=True)
        filter = st.sidebar.radio('Apply operation:', ['histogram_equalization','Clahe', 'Low light enhancement', 'white balance'])
        if filter == 'histogram_equalization' :
            image_enhance=equalize_hist(image)
            st.image(image_enhance, width=400)
        if filter == 'Clahe' :
            clipLimit=st.sidebar.number_input("cliplimit", 0, 5000, 200)
            image_enhance=clahe(image,clipLimit)
            st.image(image_enhance, width=400)
        if filter == 'Low light enhancement' :
            image_enhance=adjust_gamma(image)
            st.image(image_enhance, width=400)
        if filter ==  'white balance':
            wb = st.sidebar.radio('Apply operation:', ['histogram_equalization','Clahe'])

            image_enhance=white_balance(image)
            st.image(image_enhance, width=400)