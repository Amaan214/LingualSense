# 🌐 LingualSense

**LingualSense** is a language detection and sentiment analysis app powered by deep learning. It supports multiple input methods and provides accurate sentiment predictions across multiple languages.

## 🚀 Features

- Detects the language of the given text using FastText.
- Predicts sentiment using a pre-trained GRU-based deep learning model.
- Supports input via text box or file upload.
- Optionally download results as CSV.
- Interactive UI built with Streamlit.
- Expandable insights for sentence-level sentiment.

## 🧠 Model Architecture

The app uses a **Gated Recurrent Unit (GRU)** model trained on multilingual text data for robust sentiment prediction. The model is pre-trained and not included in the repository due to file size limits.

---

## 📥 Download GRU Model

To run the app, you need to download the GRU model file manually:

🔗 **[Download gru.h5 from Google Drive](https://drive.google.com/drive/folders/1YZOwMSYzcLUY4Os9qBDU6W_90gZ9hgy7?usp=sharing)**  
💾 Save it in the project root directory (`LingualSense/gru.h5`)

> Replace `YOUR_FILE_ID` with your actual Google Drive file ID, or use the full shared link if you prefer.

---

## 🛠️ Installation

```bash
git clone https://github.com/Amaan214/LingualSense.git
cd LingualSense
pip install -r requirements.txt
