import nltk
import random
import string
import warnings
import streamlit as st
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Suppress warnings
warnings.filterwarnings('ignore')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
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

# Predefined query-response pairs
query_response_pairs = {
    "What is the main purpose of this document?": "This document provides the terms and conditions of your Churchill motor insurance policy.",
    "How can I find my policy number?": "Your policy number is located on your policy schedule and any correspondence we've sent you.",
    "What is covered under comprehensive insurance?": "Comprehensive insurance covers damage to your car, third-party liability, and additional benefits as specified in your policy schedule.",
    "What are the exclusions of this policy?": "Exclusions include intentional damage, using the vehicle for unlawful purposes, and driving under the influence of alcohol or drugs.",
    "How do I file a claim?": "To file a claim, contact our claims team at 0345 603 3551 and provide details of the incident.",
    "How can I contact customer service?": "You can contact our customer service team at 0345 603 3551.",
    "What are the legal requirements for driving abroad?": "Your policy provides the minimum cover required by law in the European Union. Check your policy schedule for full details.",
    "What should I do if I change my address?": "If you change your address, contact our customer service team at 0345 603 3551 to update your policy details.",
    "Does the policy cover breakdown assistance?": "Breakdown assistance is an optional cover that you can add to your policy. Check your policy schedule to see if it is included.",
    "How do I make a claim?": "To make a claim, call our claims team on 0345 603 3551 and follow the instructions provided.",
    "What should I do if my car is stolen?": "If your car is stolen, report the theft to the police and then contact our claims team at 0345 603 3551.",
    "How can I get a copy of my policy document?": "You can request a copy of your policy document by calling our customer service team at 0345 603 3551.",
    "What should I do if I lose my insurance certificate?": "If you lose your insurance certificate, contact our customer service team at 0345 603 3551 to request a replacement.",
    "Are personal belongings covered by the policy?": "Personal belongings in the car are covered up to a certain limit as specified in your policy schedule.",
    "How can I reduce my premium?": "You can reduce your premium by increasing your voluntary excess, maintaining a no-claims discount, and ensuring your vehicle has security features.",
    "What is a no-claims discount?": "A no-claims discount is a reduction in your premium for each year you do not make a claim. Details are in your policy schedule.",
    "How do I update my policy details?": "You can update your policy details by contacting our customer service team at 0345 603 3551.",
    "What is the policy renewal process?": "We will send you a renewal notice before your policy expires. Follow the instructions in the notice to renew your policy.",
    "How do I add another driver to my policy?": "To add another driver to your policy, contact our customer service team at 0345 603 3551 with the details of the additional driver.",
    "What is the excess on my policy?": "The excess is the amount you have to pay towards a claim. The specific amount is detailed in your policy schedule.",
    "How do I report a windscreen claim?": "For windscreen claims, call our windscreen helpline at 0345 602 3366.",
    "Does the policy cover legal expenses?": "Legal expenses cover is an optional add-on. Check your policy schedule to see if it is included.",
    "What should I do if I miss a premium payment?": "If you miss a premium payment, contact our customer service team immediately to avoid cancellation of your policy.",
    "Does the policy cover personal injury?": "Personal injury cover is included up to the limits specified in your policy schedule.",
    "Can I get temporary insurance?": "Temporary insurance is not typically offered. Contact our customer service team for advice on your specific needs.",
    "How do I update my payment details?": "To update your payment details, contact our customer service team at 0345 603 3551.",
    "What happens if I modify my vehicle?": "Modifications must be declared and may affect your cover. Contact our customer service team to discuss any modifications.",
    "How do I check my claim status?": "You can check the status of your claim by calling our claims team at 0345 603 3551.",
    "What happens if my car is written off?": "If your car is written off, we will pay the market value of the car at the time of the loss, subject to policy terms and conditions.",
    "Does the policy cover rental car expenses?": "Rental car expenses are covered if you have chosen this optional add-on. Check your policy schedule for details.",
    "What is the process for policy cancellation?": "You can cancel your policy by contacting our customer service team at 0345 603 3551. Cancellation terms and fees may apply.",
}

# Vectorizer
def response(user_response, pdf_text, query_response_pairs):
    chatbot_response = ''
    user_response = user_response.lower()

    if user_response in query_response_pairs:
        return query_response_pairs[user_response]

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

    # Upload PDF
    pdf_file = st.file_uploader("Upload a PDF", type="pdf")

    if pdf_file is not None:
        # Extract text from the PDF document
        pdf_text = extract_text_from_pdf(pdf_file)
        if pdf_text:
            pdf_text = preprocess_text(pdf_text)
            st.write("PDF Loaded Successfully!")

            st.subheader("Ask a Question:")
            user_question = st.text_input("You: ")

            if st.button("Ask"):
                if user_question.strip() != '':
                    if greeting(user_question) is not None:
                        st.text_area("Bot:", value=greeting(user_question))
                    else:
                        response_text = response(user_question, pdf_text, query_response_pairs)
                        st.text_area("Bot:", value=response_text)
                else:
                    st.warning("Please enter a question.")
        else:
            st.error("Failed to extract text from the PDF.")

if __name__ == "__main__":
    main()
