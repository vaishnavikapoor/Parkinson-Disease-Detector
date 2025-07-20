# Parkinson’s Disease Prediction App

This web app predicts the likelihood of Parkinson's Disease based on voice features using a Decision Tree model trained on the UCI dataset.

### Features
- Trained on 22 biomedical voice features
- Uses DecisionTreeClassifier
- Accuracy: ~90%+
- Includes interactive visualizations (heatmap, scatterplot, histogram)
- Built with Streamlit

### Live Demo
[👉 Click here to try the app](https://parkinson-disease-detector-ak65caesfqtphazbzmuoxf.streamlit.app/)

### 📁 Project Structure
- `parkinsons_app.py` – Streamlit application
- `parkinsons_detector.pkl` – Pretrained model
- `parkinsons data.csv` – Dataset
- `parkinsons_notebook.ipynb` – Model development notebook

### Libraries Used
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- streamlit
