# you need to install all these in your terminal
# pip install streamlit
# pip install scikit-learn
# pip install python-docx
# pip install PyPDF2


import streamlit as st
import pickle
import docx  # Extract text from Word file
import PyPDF2  # Extract text from PDF
import re

# Load pre-trained model and TF-IDF vectorizer
svc_model = pickle.load(open('models/clf.pkl', 'rb')) 
tfidf = pickle.load(open('models/tfidf.pkl', 'rb'))  
le = pickle.load(open('models/encoder.pkl', 'rb')) 


# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText


# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text


# Function to extract text from TXT with explicit encoding handling
def extract_text_from_txt(file):
    # Try using utf-8 encoding for reading the text file
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        # In case utf-8 fails, try 'latin-1' encoding as a fallback
        text = file.read().decode('latin-1')
    return text


# Function to handle file upload and extraction
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text


# Function to predict the category of a resume
def pred(input_resume):
    # Preprocess the input text (e.g., cleaning, etc.)
    cleaned_text = cleanResume(input_resume)

    # Vectorize the cleaned text using the same TF-IDF vectorizer used during training
    vectorized_text = tfidf.transform([cleaned_text])

    # Convert sparse matrix to dense
    vectorized_text = vectorized_text.toarray()

    # Prediction
    predicted_category = svc_model.predict(vectorized_text)

    # get name of predicted category
    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0]  # Return the category name



# Define main app layout
def main():
    st.set_page_config(page_title="Smart Resume Analyzer", page_icon="🧠", layout="wide")

    st.title("📄 AI-Powered Job Role Predictor")
    st.markdown("Upload a resume file (PDF, DOCX, or TXT), and we'll tell you the most likely job category it fits.")

    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded_file = st.file_uploader("📤 Upload your Resume", type=["pdf", "docx", "txt"])

    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135768.png", width=150)

    st.markdown("---")

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("✅ Resume text extracted successfully.")

            with st.expander("📄 Show Extracted Resume Text"):
                st.text_area("Resume Content", resume_text, height=300)

            st.subheader("🔍 Recommended Job Category")
            category = pred(resume_text)

            emoji_map = {
                "Data Science": "📊",
                "Software Engineer": "💻",
                "Health & Fitness": "🏋️‍♂️",
                "Mechanical Engineer": "⚙️",
                "Civil Engineer": "🏗️",
                "Electrical Engineer": "🔌",
                "Other": "🧩"
            }

            emoji = emoji_map.get(category, "🧩")
            st.markdown(f"### {emoji} **{category}**")

            # Optionally add download button
            st.download_button("⬇️ Download Result Summary",
                               data=f"Resume Category: {category}\n\nResume Extract:\n{resume_text[:1000]}...",
                               file_name="resume_analysis.txt")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()