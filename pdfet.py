import nltk
import random
import string
import warnings
import io
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Suppress warnings
warnings.filterwarnings('ignore')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    with open(pdf_file, "rb") as f:
        reader = PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    return text

# Initialize NLTK resources
nltk.download("punkt")
nltk.download("wordnet")

# Load NLTK Lemmatizer
lemmer = nltk.stem.WordNetLemmatizer()

# Preprocessing functions
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "nods", "hi there", "hello", "I am glad! you are talking to me"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Vectorizer
def response(user_response, pdf_text):
    chatbot_response = ''
    sent_tokens = nltk.sent_tokenize(pdf_text)
    sent_tokens.append(user_response)
    tfidf_vectorizer = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf_matrix[-1], tfidf_matrix)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        chatbot_response = chatbot_response + "I am sorry! I don't understand you"
        return chatbot_response
    else:
        chatbot_response = chatbot_response + sent_tokens[idx]
        return chatbot_response

# Streamlit Web Application
def main():
    st.title("PDF-based Chatbot")

    # Specify the document location
    pdf_location = "C:\\Users\\HP\\OneDrive\\Desktop\\NOTES\\Compiler\\UNIT 5\\CD UNIT 5.pdf"

    # Extract text from the PDF document
    pdf_text = extract_text_from_pdf(pdf_location)
    pdf_text = preprocess_text(pdf_text)

    st.write("PDF Loaded Successfully!")

    st.subheader("Ask a Question:")
    user_question = st.text_input("You: ")

    if st.button("Ask"):
        if user_question.strip() != '':
            if greeting(user_question) is not None:
                st.text_area("Bot:", value=greeting(user_question))
            else:
                response_text = response(user_question, pdf_text)
                st.text_area("Bot:", value=response_text)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
