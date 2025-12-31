"""
Inference module for counting wheat heads in field images using a DeepLabV3+ semantic
segmentation model trained on the GWFSS dataset.

The model performs multi-class segmentation (Background, Leaf, Stem, Head) to accurately
distinguish wheat heads from other plant organs, then uses connected component analysis
to count individual heads.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
from scipy import ndimage
from skimage.feature import peak_local_max

# ImageNet normalisation constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Mask colours for visualization
MASK_COLORS = [
    (0, 0, 0),          # Background: black
    (214, 255, 50),     # Leaf: yellow-green
    (50, 132, 255),     # Stem: blue
    (50, 255, 132),     # Head: cyan-green
]

class GWFSSModel:
    def __init__(self, model_path, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        
        # Load model architecture
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights=None,
            in_channels=3,
            classes=4,
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    
    def preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)

    def predict(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        
        image_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            logits = self.model(image_tensor)
        
        predictions = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        return predictions

    def count_heads(self, predictions, min_distance=15):
        head_mask = (predictions == 3).astype(np.uint8)
        
        if head_mask.sum() == 0:
            return 0
        
        # Compute distance transform
        distance = ndimage.distance_transform_edt(head_mask)
        
        # Find local peaks (head centers)
        coords = peak_local_max(distance, min_distance=min_distance, labels=head_mask)
        
        # Count the peaks
        num_heads = len(coords)
        
        return num_heads

    def create_colored_mask(self, predictions):
        h, w = predictions.shape
        mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in enumerate(MASK_COLORS):
            mask_rgb[predictions == class_id] = color
        
        return Image.fromarray(mask_rgb)

    def overlay_mask(self, image, predictions, alpha=0.5, heads_only=True):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.Resampling.BILINEAR)
        
        # Create mask
        h, w = predictions.shape
        mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        if heads_only:
            # Only highlight heads
            mask_rgb[predictions == 3] = (50, 255, 132)
        else:
            # Show all classes
            for class_id, color in enumerate(MASK_COLORS):
                mask_rgb[predictions == class_id] = color
        
        mask_img = Image.fromarray(mask_rgb)
        overlay = Image.blend(image.convert('RGB'), mask_img, alpha)
        return overlay

    def predict_and_overlay(self, image, alpha=0.5, heads_only=True):
        predictions = self.predict(image)
        overlay = self.overlay_mask(image, predictions, alpha=alpha, heads_only=heads_only)
        return overlay

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path> [model_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "cache/02_dice_stem.pth"
    
    print(f"Loading model from {model_path}...")
    model = GWFSSModel(model_path)
    
    print(f"Processing image: {image_path}")
    image = Image.open(image_path)
    predictions = model.predict(image)
    
    # Count heads
    num_heads = model.count_heads(predictions)
    print(f"\n🌾 {num_heads} heads detected!")
    
    # Create visualisations
    print("\nGenerating visualisations...")
    overlay_heads = model.overlay_mask(image, predictions, alpha=0.5, heads_only=True)
    overlay_all = model.overlay_mask(image, predictions, alpha=0.5, heads_only=False)
    
    # Save outputs
    output_heads = image_path.rsplit('.', 1)[0] + '_heads_only.png'
    output_all = image_path.rsplit('.', 1)[0] + '_all_classes.png'
    
    overlay_heads.save(output_heads)
    overlay_all.save(output_all)
    
    print(f"✓ Saved head overlay to: {output_heads}")
    print(f"✓ Saved full segmentation to: {output_all}")