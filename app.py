import pickle
import re
from nltk.stem import PorterStemmer
import nltk

# Load the saved model, LabelEncoder, and other components
log_reg_model = pickle.load(open('logistic_regresion.pkl', 'rb'))  # Assuming the model was saved as log_reg_model.pkl
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
lb = pickle.load(open('label_encoder.pkl', 'rb'))

# Download NLTK stopwords
nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

# Preprocessing function to clean user input text
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

# Prediction function
def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])  # Transform input using the same vectorizer used in training

    # Predict emotion using the logistic regression model
    predicted_label = log_reg_model.predict(input_vectorized)
    predicted_emotion = lb.inverse_transform([predicted_label])[0]  # Get the emotion label from the encoder

    return predicted_emotion

# Loop to take messages and make predictions
while True:
    # Take user input for message
    user_input = input("Enter a message (or 'quit' to stop): ")
    
    if user_input.lower() == 'quit':
        print("Exiting the emotion detection model.")
        break

    # Get the predicted emotion
    predicted_emotion = predict_emotion(user_input)
    
    # Output the predicted emotion
    print(f"Predicted Emotion: {predicted_emotion}")
