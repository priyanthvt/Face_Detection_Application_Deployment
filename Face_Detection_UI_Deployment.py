import streamlit as st
from PIL import Image
import torch

import sys
sys.path.append("yolov5")  # Tell Python to look inside the yolov5 folder

# from models.common import DetectMultiBackend  # Import from yolov5

from models.experimental import attempt_load
# from utils.general import non_max_suppression, scale_coords
from yolov5.utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

import torchvision.transforms as transforms
import torch

import cv2
import numpy as np

st.set_page_config(
    page_title="YOLOv5 Face Detection",
    layout="centered",
    initial_sidebar_state="auto"
)

def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# @st.cache_resource
# def load_model():
#     # model_path = r"E:\Priyanth\AIML\Project\FaceDetection\FaceDetectionVScode\yolov5\runs\train\face_detect\weights\best.pt"
#     model_path = r"model\best.pt"
#     model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=True)
#     return model

# @st.cache_resource
# def load_model():
#     model_path = "model/best.pt"  # Update with your path if needed
#     model = DetectMultiBackend(model_path, device='cpu')  # or 'cuda' if you're using GPU
#     return model

@st.cache_resource
def load_model():
    model_path = "model/best.pt"
    device = select_device('cpu')  # Or 'cuda' if GPU is available
    model = attempt_load(model_path, map_location=device)
    model.eval()
    return model

model = load_model()

if "page" not in st.session_state:
    st.session_state.page = "home"
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "results" not in st.session_state:
    st.session_state.results = None

def home():
    set_background("https://images.unsplash.com/photo-1593642634443-44adaa06623a?ixlib=rb-4.0.3&auto=format&fit=crop&w=1350&q=80")

    st.markdown(
    """
    <h1 style='color: white;'>Face Detection with YOLOv5</h1>
    <p style='color: white; font-size: 18px;'>
        Welcome to the <strong>YOLOv5 Face Detection App</strong>!<br>
        This application allows you to upload an image and detect faces using a custom-trained YOLOv5 model.
        <br><br>
        Accurate Face Detection<br>
        Fast Inference<br>
        Easy Upload Interface
    </p>
    <hr style='border: 1px solid white;'>
    """,
    unsafe_allow_html=True
    )

    if st.button("Start Detecting Faces"):
        st.session_state.page = "detect"
        st.rerun()

def detect_faces():
    set_background("https://images.unsplash.com/photo-1519389950473-47ba0277781c?auto=format&fit=crop&w=1350&q=80")

    st.markdown("<h1 style='color: #E0FFFF;'>Upload an Image</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #E0FFFF;'>Please upload an image (JPG, JPEG, or PNG) to detect faces.</p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.session_state.uploaded_image = image

        # if st.button("Run Face Detection"):
        #     with st.spinner("Detecting faces..."):
        #         # results = model(image)
        #         image_tensor = transforms.ToTensor()(image).unsqueeze(0)  # Convert to [1, 3, H, W]
        #         image_tensor = image_tensor * 255  # Scale to [0,255] range if needed
        #         image_tensor = image_tensor.to(torch.float32)

        #         # Run inference
        #         with torch.no_grad():
        #             pred = model(image_tensor)[0]
        #             pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

        #         # Save results to session
        #         st.session_state.results = pred
        #         # st.session_state.results = results
        #         st.session_state.page = "results"
        #         st.rerun()

        if st.button("Run Face Detection"):
            with st.spinner("Detecting faces..."):
                # Convert image to tensor
                image_tensor = transforms.ToTensor()(image).unsqueeze(0) * 255  # [0,1] -> [0,255]
                image_tensor = image_tensor.to(torch.float32)
        
                # Run inference
                with torch.no_grad():
                    pred = model(image_tensor)[0]
                    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)
        
                # Save results to session
                st.session_state.results = pred
                st.session_state.page = "results"
                st.rerun()

    if st.button("Back to Home"):
        st.session_state.page = "home"
        st.rerun()

def results_page():
    set_background("https://images.unsplash.com/photo-1498050108023-c5249f4df085?auto=format&fit=crop&w=1350&q=80")
    st.title("Detection Results")

    if st.session_state.results is not None and st.session_state.uploaded_image is not None:
        results = st.session_state.results
        # num_faces = results.xyxy[0].shape[0]
        # st.image(results.render()[0], caption=f"Detected {num_faces} face(s)", use_container_width=True)

        # Convert PIL image to OpenCV format
        image = np.array(st.session_state.uploaded_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        detections = results[0]
        num_faces = len(detections)
        
        # Draw boxes
        for *xyxy, conf, cls in detections:
            xyxy = [int(x.item()) for x in xyxy]
            cv2.rectangle(image, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            cv2.putText(image, f'{conf:.2f}', (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Convert back to RGB and display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption=f"Detected {num_faces} face(s)", use_container_width=True)


        
        if num_faces > 0:
            st.success(f"Successfully detected {num_faces} face(s).")
        else:
            st.warning("No detection results found.")
    else:
        st.warning("No detection results found.")
        st.session_state.page = "detect"
        st.rerun()

    st.markdown("---")
    st.markdown("**Thank you for using the app!**")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Detect Another Image"):
            st.session_state.page = "detect"
            st.session_state.uploaded_image = None
            st.session_state.results = None
            st.rerun()
    with col2:
        if st.button("Go to Home"):
            st.session_state.page = "home"
            st.session_state.uploaded_image = None
            st.session_state.results = None
            st.rerun()

if st.session_state.page == "home":
    home()
elif st.session_state.page == "detect":
    detect_faces()
elif st.session_state.page == "results":
    results_page()





