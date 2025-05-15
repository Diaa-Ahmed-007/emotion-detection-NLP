import pickle
import re
from nltk.stem import PorterStemmer
import nltk

log_reg_model = pickle.load(open('logistic_regresion.pkl', 'rb'))  # Assuming the model was saved as log_reg_model.pkl
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
lb = pickle.load(open('label_encoder.pkl', 'rb'))

nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text]) 

    predicted_label = log_reg_model.predict(input_vectorized)
    predicted_emotion = lb.inverse_transform([predicted_label])[0] 

    return predicted_emotion

while True:

    user_input = input("Enter a message (or 'quit' to stop): ")
    
    if user_input.lower() == 'quit':
        print("Exiting the emotion detection model.")
        break

    predicted_emotion = predict_emotion(user_input)
    
    print(f"Predicted Emotion: {predicted_emotion}")
