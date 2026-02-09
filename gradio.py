# app.py
import gradio as gr
import pickle
from sentence_transformers import SentenceTransformer
from tensorflow.keras.models import load_model

# Load model and encoder
model = load_model('disease_classifier.h5')
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
clinicalbert = SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT')

def predict_disease(symptoms):
    X = clinicalbert.encode([symptoms])
    prediction = model.predict(X, verbose=0)
    predicted_class = prediction.argmax()
    confidence = prediction.max()
    disease = label_encoder.inverse_transform([predicted_class])[0]
    return f"**Disease:** {disease}\n\n**Confidence:** {confidence:.1%}"

demo = gr.Interface(
    fn=predict_disease,
    inputs=gr.Textbox(label="Enter your symptoms", placeholder="e.g., I have blackheads and pimples"),
    outputs=gr.Markdown(label="Prediction"),
    title="🏥 Disease Prediction System",
    description="Enter your symptoms to get a disease prediction"
)

demo.launch()