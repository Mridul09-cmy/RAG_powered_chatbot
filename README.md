
## RAG-based Chatbot

This project implements a chatbot capable of answering questions based on the content of a PDF document. The chatbot utilizes Natural Language Processing (NLP) techniques to process user queries and retrieve relevant information from the PDF document.

### Features:

- **PDF Text Extraction**: The chatbot extracts text from a specified PDF document.
- **Preprocessing**: The extracted text is preprocessed to lower case and remove punctuation for better analysis.
- **TF-IDF Vectorization**: The chatbot employs TF-IDF (Term Frequency-Inverse Document Frequency) vectorization to convert text data into numerical vectors.
- **Cosine Similarity**: It calculates the cosine similarity between user queries and sentences in the PDF document to find the most relevant response.
- **Streamlit Web Application**: The chatbot is deployed as a web application using Streamlit, allowing users to interact with it via a user-friendly interface.
- **Greeting Recognition**: The chatbot recognizes common greetings and responds accordingly.
- **Natural Language Understanding**: The chatbot attempts to understand user queries and provides appropriate responses based on the content of the PDF document.

### Usage:

1. Upload a PDF document.
2. Ask questions related to the content of the PDF.
3. Receive responses based on the extracted information from the PDF.

### Dependencies:

- Python 3.x
- NLTK (Natural Language Toolkit)
- PyPDF2
- Streamlit
- scikit-learn

### How to Run:

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Streamlit web application using `streamlit run app.py`.
4. Upload a PDF document and start asking questions!
