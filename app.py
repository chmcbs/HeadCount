import gradio as gr
import os
from inference import GWFSSModel
from PIL import Image
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from huggingface_hub import hf_hub_download

# Download model from Hugging Face
print("Downloading model from Hugging Face...")
MODEL_PATH = hf_hub_download(repo_id="chmcbs/HeadCount", filename="model.pth")
print("✓ Model downloaded successfully")

# Load model
print("Loading model...")
model = GWFSSModel(MODEL_PATH)
print("✓ Model loaded successfully")

def process_image(image):
    if image is None:
        return "", None
    
    try:
        predictions = model.predict(image)
        num_heads = model.count_heads(predictions)
        
        # Visualise detected peaks
        head_mask = (predictions == 3).astype(np.uint8)
        distance = ndimage.distance_transform_edt(head_mask)
        coords = peak_local_max(distance, min_distance=15, labels=head_mask)
        
        # Create overlay with peak markers
        overlay = model.overlay_mask(image, predictions, alpha=0.5, heads_only=True)
        overlay_np = np.array(overlay)
        for y, x in coords:
            # Draw a small red circle at each detected peak
            overlay_np[max(0,y-3):y+4, max(0,x-3):x+4] = [255, 0, 0]
        
        overlay = Image.fromarray(overlay_np)
        count_message = f"### 🌾 {num_heads} heads detected!"
        return count_message, overlay
    except Exception as e:
        return f"Error: {str(e)}", None

# Get example images
example_images = []
if os.path.exists("examples"):
    example_files = sorted([f for f in os.listdir("examples") 
                           if f.endswith(('.jpg', '.jpeg', '.png'))])[:5]
    example_images = [os.path.join("examples", f) for f in example_files]

# Create Gradio interface
with gr.Blocks(title="HeadCount") as demo:
    gr.Markdown("# 🌾 HeadCount: Automated Wheat Head Counter")
    gr.Markdown("Upload an image to automatically detect and count wheat heads.")
    
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        
        with gr.Column():
            overlay_output = gr.Image(label="Segmentation Overlay")
            generate_btn = gr.Button("Generate", variant="primary")
    
    with gr.Row():
        with gr.Column():
            if example_images:
                gr.Markdown("### Example Images")
                gr.Examples(examples=example_images, inputs=image_input)
        
        with gr.Column():
            head_count_output = gr.Markdown(value="")
    
    generate_btn.click(
        fn=process_image,
        inputs=image_input,
        outputs=[head_count_output, overlay_output]
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)