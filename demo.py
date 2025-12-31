import gradio as gr
import os
from inference import GWFSSModel
from PIL import Image
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max

# Model configuration
MODEL_PATH = "cache/02_dice_stem.pth"
VALIDATION_DIR = "validation/images"

# Load model once at startup
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

# Get example images from validation set
example_images = []
if os.path.exists(VALIDATION_DIR):
    example_files = sorted([f for f in os.listdir(VALIDATION_DIR) 
                           if f.endswith(('.jpg', '.jpeg', '.png'))])[:5]
    example_images = [os.path.join(VALIDATION_DIR, f) for f in example_files]

# Create Gradio interface
with gr.Blocks(title="HeadCount") as demo:
    gr.Markdown("# 🌾 HeadCount")
    gr.Markdown("### A semantic segmentation model for counting wheat heads in field images.")
    gr.Markdown("- Designed for yield estimation, flowering time detection, and field maturity assessment")
    gr.Markdown("- Best results with overhead imagery under diffuse lighting")
    gr.Markdown("- Performance may vary with harsh lighting or heavily overlapping heads")
    
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        
        with gr.Column():
            overlay_output = gr.Image(label="Segmentation Overlay")
            generate_btn = gr.Button("Generate", variant="primary")
    
    with gr.Row():
        with gr.Column():
            if example_images:
                gr.Examples(examples=example_images, inputs=image_input)
        
        with gr.Column():
            head_count_output = gr.Markdown(value="")
    
    generate_btn.click(
        fn=process_image,
        inputs=image_input,
        outputs=[head_count_output, overlay_output]
    )

    gr.Markdown("<div style='text-align: center'>Powered by <strong><a href='https://huggingface.co/chmcbs/HeadCount' target='_blank'>chmcbs/HeadCount</a></strong></div>")

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)