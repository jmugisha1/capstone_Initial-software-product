# Disease Prediction System

A machine learning-based disease prediction system that uses ClinicalBERT embeddings and an Artificial Neural Network (ANN) classifier to predict diseases from symptom descriptions.

## Description

This project uses a hybrid approach combining natural language processing and deep learning to predict diseases based on user-entered symptoms. The system architecture consists of:

- **ClinicalBERT**: Transforms symptom text into medical domain-specific embeddings (768-dimensional vectors)
- **Artificial Neural Network (ANN)**: Multi-layer classification model that predicts disease from embeddings
- **Gradio Interface**: Interactive web interface for user interaction

**Model Architecture:**
- Input: ClinicalBERT embeddings (768 dimensions)
- Hidden Layer 1: 128 neurons with ReLU activation
- Hidden Layer 2: 64 neurons with ReLU activation
- Output Layer: Softmax activation for multi-class classification

**Model Performance:**
- Accuracy: 65.1%
- F1-Score: 61.1%

## GitHub Repository

[https://github.com/yourusername/disease-prediction](https://github.com/yourusername/disease-prediction)

## Setup Instructions

### Prerequisites
- Python 3.8+
- pip

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/disease-prediction.git
cd disease-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download model files**
Place these files in the project root:
- `disease_classifier.h5`
- `label_encoder.pkl`

### Run the Application
```bash
python app.py
```

The app will be available at `http://localhost:7860`

## Requirements
```txt
gradio==4.14.0
tensorflow==2.15.0
sentence-transformers==2.2.2
pandas==2.1.4
scikit-learn==1.3.2
numpy==1.24.3
```

## Deployment Plan

### Option 1: Hugging Face Spaces (Recommended)

1. **Create a Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Select SDK: **Gradio**

2. **Upload files**
```
   disease-prediction-space/
   ├── app.py
   ├── disease_classifier.h5
   ├── label_encoder.pkl
   └── requirements.txt
```

3. **Push to Space**
```bash
   git clone https://huggingface.co/spaces/yourusername/disease-prediction
   cd disease-prediction
   cp app.py disease_classifier.h5 label_encoder.pkl requirements.txt .
   git add .
   git commit -m "Initial commit"
   git push
```

4. Space will automatically build and deploy

### Option 2: Render

1. Create account at [render.com](https://render.com)
2. Connect GitHub repository
3. Create **Web Service**
4. Configure:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
5. Deploy

### Option 3: Local Deployment with Public URL
```bash
python app.py --share
```

## Project Structure
```
disease-prediction/
├── app.py                      # Gradio interface
├── disease_classifier.h5       # Trained ANN model
├── label_encoder.pkl           # Label encoder for disease names
├── requirements.txt            # Python dependencies
├── train_model.ipynb          # Model training notebook
├── diseases.csv               # Training dataset
└── README.md                  # Documentation
```

## Usage

1. Enter symptoms in natural language (e.g., "I have blackheads and pimples")
2. Click Submit
3. View predicted disease name and confidence score

## Model Training Pipeline

The complete training process (see `modeling.ipynb`):

1. **Data Preprocessing**: Clean and prepare symptom-disease pairs
2. **Text Embedding**: Convert symptoms to vectors using ClinicalBERT
3. **ANN Training**: Train neural network classifier on embeddings
4. **Evaluation**: Calculate accuracy and F1-score metrics

## Technical Details

**ClinicalBERT** is used exclusively for embedding generation, converting raw symptom text into meaningful numerical representations that capture medical context.

**ANN Classifier** performs the actual disease classification using the embeddings as input features. The model uses sparse categorical cross-entropy loss and Adam optimizer.

## License

MIT License

## Author

Your Name - [GitHub](https://github.com/yourusername)

## Acknowledgments

- ClinicalBERT by Emily Alsentzer for medical text embeddings
- TensorFlow/Keras for neural network implementation
- Gradio for web interface framework
