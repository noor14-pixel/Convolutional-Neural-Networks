Overview
A three-part deep dive into Convolutional Neural Networks, covering everything from implementing convolutions from scratch to GradCAM-based interpretability and PCA-CNN connections.

Dataset: CIFAR-10 (Parts 1 & 2), Fashion-MNIST (Part 3)
Framework: PyTorch + NumPy
Final Test Accuracy (Part 2): 82.38% with only 289,194 parameters


Structure
PA1/
├── Part1_PA1.ipynb        # CNN & Bottleneck from Scratch
├── Part2_PA1.ipynb        # Efficient CNN + GradCAM
├── Part3_PA1.ipynb        # PCA Meets CNNs
└── README.md

Part 1: CNN and Bottleneck from Scratch
Implemented core CNN components using NumPy only (no PyTorch autograd):

2D Convolution Layer — forward and backward pass from scratch, including gradient computation w.r.t. input, weights, and biases
Cross-Entropy Loss — numerically stable implementation using the log-sum-exp trick
Standard CNN — 3 conv blocks (32→64→128 channels) + 2 FC layers for CIFAR-10
Bottleneck Block — 1×1 squeeze → 3×3 conv → 1×1 expand with residual connection (~8.5× fewer multiplications than standard conv)
Training loop with Adam optimizer, accuracy tracking, and confusion matrix visualization


Part 2: Efficient CNN + GradCAM
Designed an efficient CNN under a strict 1M parameter budget and used GradCAM to interpret its predictions.
Architecture
Layer GroupChannelsOutput SizeConv Block 1 (×2)3 → 3216×16Conv Block 2 (×2)32 → 648×8Conv Block 3 (×2)64 → 1284×4Global Avg Pool + FC128 → 10—

BatchNorm + ReLU + Dropout (0.3) after each block
Total parameters: 289,194 (well under the 1M budget)

Results
EpochTrain AccTest Acc573.4%74.5%1079.1%80.3%1581.8%82.23%Best—82.38%
GradCAM Visualizations
Implemented visualize_gradcam and visualize_gradcam_multiple_layers using PyTorch forward/backward hooks to generate class-discriminative heatmaps. Visualizations show a clear progression from edge detection in early layers to object-level attention in deeper layers.

Part 3: PCA Meets CNNs
Explored the connection between classical PCA and learned CNN representations on Fashion-MNIST:

Activation extraction with forward hooks across conv and FC layers
PCA latent space analysis — 2D scatter plots showing increasing class separability in deeper layers
PCA filter initialization — initialized conv1 filters with PCA eigenvectors computed from image patches, compared training dynamics against random initialization
Ablation study — varied number of PCA components used to reconstruct input images and measured the effect on CNN accuracy


Key Findings

A CNN with fewer than 300K parameters can exceed 82% accuracy on CIFAR-10 using careful architectural choices: gradual channel expansion, Global Average Pooling instead of large FC layers, and BatchNorm + Dropout
GradCAM reveals that early layers respond to low-level edges while deeper layers attend to semantically meaningful object regions
CNNs are surprisingly robust to aggressive PCA input compression — performance degrades gracefully even with very few components
PCA eigenvectors of image patches visually resemble learned conv filters (edge detectors, Gabor-like patterns), showing the connection between unsupervised and supervised feature learning


Setup
bashpip install torch torchvision numpy matplotlib seaborn scikit-learn opencv-python tqdm
Run each notebook in order. CIFAR-10 and Fashion-MNIST download automatically via torchvision.datasets.

Requirements

Python 3.8+
PyTorch 2.x
CUDA-compatible GPU recommended (CPU works but is significantly slower)
