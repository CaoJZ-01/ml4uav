# Python In-built packages
from pathlib import Path
import PIL
import io

# External packages
import streamlit as st

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLO Algorithm",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Object Detection using YOLO Algorithm")

# Sidebar
st.sidebar.header("ML Model Config")

task_selection = st.sidebar.radio("Select Task", ['Drone Detection', 'Powerline Detection'])

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image")
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image")
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image')
        else:
            detect_button_clicked = st.sidebar.button('Detect Objects')
            if detect_button_clicked or 'res_plotted' in st.session_state:
                if detect_button_clicked:
                    res = model.predict(uploaded_image, conf=confidence)
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.session_state['res_plotted'] = res_plotted
                    st.session_state['boxes'] = boxes
                else:
                    res_plotted = st.session_state['res_plotted']
                    boxes = st.session_state['boxes']

                st.image(res_plotted, caption='Detected Image')

                # Convert the detected image to bytes for download
                buf = io.BytesIO()
                detected_image = PIL.Image.fromarray(res_plotted)
                detected_image.save(buf, format="PNG")
                byte_im = buf.getvalue()

                st.download_button(
                    label="Download object-detected image",
                    data=byte_im,
                    file_name="detected_image.png",
                    mime="image/png"
                )

                # Convert detection results to text for download
                detection_results = "\n".join([str(box.data) for box in boxes])
                st.download_button(
                    label="Download detection results",
                    data=detection_results,
                    file_name="detection_results.txt",
                    mime="text/plain"
                )

                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    source_video = st.sidebar.file_uploader(
        "Choose a video...", type=("mp4", "avi", "mov", "mkv")
    )

    if source_video is not None:
        st.video(source_video, format="video/mp4", start_time=0)
        if st.sidebar.button('Process Video'):
            try:
                helper.process_uploaded_video(source_video, confidence, model)
                st.success("Video processed successfully!")
            except Exception as ex:
                st.error("Error occurred while processing the video.")
                st.error(ex)
    else:
        st.info("Please upload a video to process.")

elif source_radio == settings.WEBCAM:
    helper.play_webcam(confidence, model)

elif source_radio == settings.RTSP:
    helper.play_rtsp_stream(confidence, model)

elif source_radio == settings.YOUTUBE:
    helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
