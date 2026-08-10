# Metaheuristic-Assisted Deep Learning for Brain Hemorrhage Detection and Classification

A deep learning system that detects and classifies brain hemorrhage from CT scan images using a Convolutional Neural Network (CNN) enhanced with Particle Swarm Optimization (PSO) for hyperparameter tuning.

## Overview

Brain hemorrhage is a life-threatening condition caused by bleeding in the brain, requiring fast and accurate diagnosis to prevent serious complications. This project implements a **Metaheuristic-Assisted Deep Learning** pipeline that:

- Uses a **CNN (MobileNetV2 backbone)** to automatically extract and classify features from brain CT scan images.
- Uses **Particle Swarm Optimization (PSO)** to optimize key hyperparameters (learning rate, number of neurons, dropout rate) instead of manual tuning.
- Classifies CT scans into two categories: **Normal** and **Hemorrhage**.

The combined CNN+PSO framework achieves higher accuracy, robustness, and generalization compared to a standalone CNN or other pretrained architectures (VGG16, ResNet50).

## Key Features

- Automated feature extraction from CT scan images using a pretrained MobileNetV2 backbone.
- Hyperparameter optimization via Particle Swarm Optimization.
- Data preprocessing pipeline: resizing, normalization, and augmentation.
- Comprehensive performance evaluation: accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.
- Comparative analysis against baseline CNN, VGG16, and ResNet50 models.

## Results

| Model     | Precision | Recall | F1-Score | Accuracy |
|-----------|-----------|--------|----------|----------|
| VGG16     | 88%       | 88%    | 88%      | 87.86%   |
| ResNet50  | 77%       | 77%    | 76%      | 76.69%   |
| **CNN+PSO** | **95%** | **95%**| **95%**  |**94.64%**|

- **Per-class accuracy (CNN+PSO):** Normal – 96%, Hemorrhage – 93%
- **ROC-AUC Score:** 0.9861
- Test set confusion matrix: 492 Normal and 779 Hemorrhage images correctly classified with minimal misclassification.

## Methodology

1. **Data Collection** – CT scan images sourced from the publicly available *Brain CT Hemorrhage* dataset (Kaggle), originally from Near East Hospital, Cyprus.
2. **Data Preprocessing** – Image resizing (224×224), normalization, and data augmentation (flip, rotation, zoom).
3. **Model Building** – A CNN built on top of a pretrained MobileNetV2 backbone (ImageNet weights) with Global Average Pooling, Batch Normalization, Dense, and Dropout layers.
4. **Baseline Training** – Train the CNN model with default hyperparameters.
5. **PSO Optimization** – Optimize learning rate, number of neurons, and dropout rate using Particle Swarm Optimization to maximize validation accuracy.
6. **Final Training & Evaluation** – Train the CNN with PSO-optimized hyperparameters and evaluate using standard classification metrics.

## Project Structure

```
├── data/                             
├── brain_hemorrhage_cnn_model.h5        
├── brain_hemorrhage_cnn_pso_model.h5    
├── cnn.py
├── cnnpso.py 
├── images/        
└── README.md
```

## Requirements

### Hardware
- Processor: AMD Ryzen 5 7520U (or equivalent), 2.80 GHz
- RAM: 16 GB (minimum recommended)

### Software
- OS: Windows 11 (or any OS supporting Python/TensorFlow)
- Python 3.x
- IDE: Visual Studio Code (or any Python IDE)

### Python Libraries
```bash
pip install numpy pandas matplotlib seaborn tensorflow keras scikit-learn opencv-python tqdm
```

Key libraries used:
- `numpy`, `pandas` – data handling
- `matplotlib`, `seaborn` – visualization (accuracy/loss curves, confusion matrix)
- `tensorflow`, `keras` – model building and training (MobileNetV2 backbone)
- `scikit-learn` – evaluation metrics (classification report, ROC-AUC, confusion matrix)
- `opencv-python`, `tqdm` – image preprocessing utilities

## How to Run

1. **Clone the repository**
   ```bash
   git clone <https://github.com/Sowjanya-100/Metaheuristic-Assisted-Deep-Learning-for-Brain-Hemorrhage-Detection-and-Classification>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**
   - Download the Brain CT Hemorrhage dataset from Kaggle.
   - Place it under `dataset/train` and `dataset/test`, each containing `Normal` and `Hemorrhage` subfolders.

4. **Preprocess the data**
   - Run the preprocessing script to resize, normalize, and store images in `preprocessed_dataset/`.

5. **Train the baseline CNN**
   - Trains a MobileNetV2-based CNN and saves it as `brain_hemorrhage_cnn_model.h5`.

6. **Run PSO Optimization**
   - Searches for optimal learning rate, neuron count, and dropout rate using Particle Swarm Optimization.

7. **Train the final CNN + PSO model**
   - Trains the CNN with optimized hyperparameters and saves it as `brain_hemorrhage_cnn_pso_model.h5`.

8. **Evaluate the model**
   - Generates classification report, confusion matrix, ROC-AUC score, and accuracy/loss curves.

## Model Architecture

- **Backbone:** MobileNetV2 (pretrained on ImageNet, frozen base)
- **Added Layers:** Global Average Pooling → Batch Normalization → Dense (ReLU) → Dropout → Dense (Sigmoid, binary output)
- **Optimizer:** Adam (learning rate tuned via PSO)
- **Loss Function:** Binary Cross-Entropy
- **Optimization Algorithm:** Particle Swarm Optimization (tunes learning rate, neuron count, dropout rate)

## Future Scope

- Extend to multi-class classification (different hemorrhage subtypes: epidural, subdural, intracerebral, subarachnoid, intraventricular).
- Incorporate more diverse and larger datasets for better generalization.
- Explore advanced architectures (e.g., transformer-based models) and hybrid optimization techniques.
- Add explainability techniques (e.g., Grad-CAM) to highlight hemorrhage regions in CT scans.
- Deploy as a real-time web/mobile application integrated with hospital or IoT-based healthcare systems.

## Authors

- Tankala Charisma
- Dokkari Sowjanya
- Kompalli Keerthi
- Teeti Prasad
- Bodnaik Tharun Kumar

**Under the guidance of:** Sri. D. Sreenu Babu, Asst. Professor, Dept. of CSE

**Institution:** Department of Computer Science and Engineering, Aditya Institute of Technology and Management (AITAM), K.Kotturu, Tekkali, Srikakulam, Andhra Pradesh

## License

This project was developed as a Bachelor of Technology final year project (2026) submitted to Jawaharlal Nehru Technological University Gurajada, Vizianagaram. Please contact the authors for reuse permissions.

## Acknowledgements

This work has been accepted for oral presentation at **WAMS 2026** (Paper ID: 437).
