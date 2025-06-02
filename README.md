# Artistic Video Style Transfer with Segmentation Guidance
## Overview
This project transforms videos by applying artistic styles (e.g., Monet’s paintings) to specific objects, such as a bear, while preserving the background and ensuring smooth transitions. It combines a U-Net model for object segmentation, a VGG19-based style transfer pipeline, and a Streamlit app for user interaction. Inspired by StyTR-2 ([StyTR-2 Paper](https://doi.org/10.48550/arXiv.2105.14576)) and InST ([InST Paper](https://doi.org/10.48550/arXiv.2211.13203)), it extends image-based style transfer to videos with object-specific styling and temporal consistency, using the DAVIS 2017 dataset ([DAVIS 2017](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip).
## Features
1. Object-Specific Styling: Applies styles only to segmented objects (e.g., a bear), preserving the background.
2. Temporal Consistency: Uses optical flow to ensure smooth video transitions, avoiding flickering.
## Project Structure
1. edai-6.ipynb: Trains a U-Net model for binary segmentation on DAVIS 2017, producing masks and training metric plots (loss, accuracy, IoU).
2. main.ipynb: Implements the style transfer pipeline, processing video frames with VGG19, applying segmentation-guided styling, and ensuring temporal consistency.
## Dataset
The DAVIS 2017 dataset includes 150 videos (60 training, 30 validation, 60 testing), each with ~50–100 frames and ground-truth segmentation masks. It’s ideal for training U-Net to segment objects and testing the style transfer pipeline.
## Methodology
### U-Net Segmentation
1. Architecture: U-Net with encoder-decoder structure, skip connections, and dropout (0.3) for regularization.
2. Training: Uses a hybrid loss (Binary Cross-Entropy + Dice) to handle class imbalance, trained on 4,209 samples for 20 epochs with early stopping.
3. Metrics: Tracks accuracy and IoU, achieving a validation IoU of ~0.3956 and accuracy of ~0.8881.
### Style Transfer with VGG19
1. Feature Extraction: Pre-trained VGG19 extracts content features (e.g., bear’s shape) and style features (e.g., Monet’s brushstrokes).
2. Loss Functions: Combines content loss (preserves structure), style loss (applies artistic style via Gram matrices), and temporal consistency loss (ensures smoothness).
3. Mask Application: U-Net masks guide styling to foreground objects, preserving the background.
### Temporal Consistency
1. Optical Flow: Farneback’s method computes motion between frames.
2. Smoothing: Warps stylized frames using optical flow to maintain coherence, reducing flickering.
## Results
- Segmentation: U-Net achieves a validation IoU of 0.3956 and accuracy of 0.8881, with evaluation metrics (loss: 0.4033, accuracy: 0.8881, IoU: 0.3660).
- Style Transfer: Produces stylized videos (e.g., bear_stylized.mp4, tennis_stylized.mp4) with objects rendered in artistic styles, as shown in output.png (bear styled like Monet’s “The Japanese Footbridge”).
- Visualization: Displays content, style, and stylized images/videos, demonstrating effective style application.
## Novelty
- Video-Specific Styling: Extends StyTR-2’s image-based transformer approach to videos, adding temporal consistency absent in StyTR-2 and InST.
- Segmentation Guidance: Uses U-Net masks to target specific objects, unlike InST’s full-image styling, enhancing precision.
## Challenges
- Class Imbalance: Small foreground objects (e.g., bear) vs. large backgrounds caused U-Net to favor background predictions, addressed with hybrid loss.
- Overfitting: High training IoU (0.9124) vs. lower validation IoU (0.3956) required early stopping and augmentation.
- Resource Constraints: Kaggle’s Tesla T4 GPUs (~16GB) faced memory warnings, managed with batch size 4 and image size 256x256.
## Applications
- Artistic Creation: Transform videos into stylized artworks for social media or exhibitions.
- Entertainment: Apply effects in films, music videos, or games.
- Research: Serve as a baseline for video style transfer and segmentation studies.
## Advantages
- Speed: Leverages pre-trained VGG19 and lightweight U-Net for fast processing.
- Precision: Segmentation ensures accurate object styling.
- Smoothness: Optical flow prevents video flickering.
- Accessibility: Streamlit app enables non-experts to use the system.
