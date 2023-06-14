# Import Library
import streamlit as st
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep
import time
import requests
from io import BytesIO
from PIL import Image
import easyocr
import pandas as pd
from transformers import AutoTokenizer, BertForSequenceClassification
import torch
from pandas import DataFrame
import firebase_admin
from firebase_admin import App, initialize_app, credentials, firestore

# Set your Instagram credentials
my_user = "scrapetesting"
my_pwd = 'do.ricard0'

# Load the model
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased', num_labels=2)
model.load_state_dict(torch.load('IndoBERT_classifier.pt'))
model.eval()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-uncased')


# Initialize Firebase
def get_firebase_app():
    try:
        firebase_admin.get_app()
    except ValueError as e:
        cred = credentials.Certificate('credential.json')
        firebase_admin.initialize_app(cred)

firebase_app = get_firebase_app()
db = firestore.client()
collection = db.collection('job-ads-bias')

def get_driver():
    options = Options()
    options.add_argument('--disable-gpu')
    options.add_argument('--headless')
    return webdriver.Chrome(executable_path=ChromeDriverManager().install(), options=options)

def classify_text(text):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors='pt')

    # Make prediction
    outputs = model(**inputs)

    # Get the predicted class
    _, predicted = torch.max(outputs.logits, 1)

    # Convert the prediction to 'bias' or 'Non-bias'
    if predicted.item() == 1:
        return 'bias'
    else:
        return 'Non-bias'


def perform_image_scraping_ocr():
    driver = get_driver()
    driver.get("https://www.instagram.com/accounts/login")
    driver.maximize_window()
    sleep(3)

    user_name = driver.find_element(By.XPATH, "//input[@name='username']")
    user_name.send_keys(my_user)
    sleep(1)

    password = driver.find_element(By.XPATH, "//input[@name='password']")
    password.send_keys(my_pwd)
    password.send_keys(Keys.RETURN)
    sleep(3)

    # Keyword to search
    keyword = "lowongan"
    driver.get("https://www.instagram.com/explore/tags/" + keyword + "/")
    time.sleep(8)
    my_images = set()

    # Get all images on the page
    images = driver.find_elements(By.XPATH, "//img[@class='x5yr21d xu96u03 x10l6tqk x13vifvy x87ps6o xh8yej3']")
    while len(my_images) < 5:
        for image in images:
            source = image.get_attribute('src')
            # Check if document exists in Firestore
            docs = collection.where('image_url', '==', source).stream()
            if len(list(docs)) == 0:
                my_images.add(source)
                # Add new document to Firestore
                collection.add({'image_url': source})
            if len(my_images) >= 5:
                break
        if len(my_images) < 5:
            driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            sleep(3)
            images = driver.find_elements(By.XPATH, "//img[@class='x5yr21d xu96u03 x10l6tqk x13vifvy x87ps6o xh8yej3']")

    driver.quit()

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en', 'id'])  # Supports English and Indonesian

    # Perform OCR on each image and save to Firestore
    docs = collection.stream()
    for doc in docs:
        doc_dict = doc.to_dict()
        if 'ocr_result' not in doc_dict:
            url = doc_dict['image_url']
            response = requests.get(url)
            image = Image.open(BytesIO(response.content))
            result = reader.readtext(image)
            text = ' '.join([item[1] for item in result])
            prediction = classify_text(text)
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            # Update document in Firestore
            doc.reference.update({'ocr_result': text, 'timestamp': timestamp, 'prediction': prediction})
            
def get_data_from_db():
    # Get data from Firestore
    doc_ref = db.collection('job-ads-bias')
    docs = doc_ref.stream()

    # Create a DataFrame
    df = DataFrame([doc.to_dict() for doc in docs])

    # Convert the 'timestamp' column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.sort_values(by='timestamp', ascending=False, inplace=True)

    return df

def main():
    st.title('Bias and Discrimination Recognition')

    # Load initial data
    df = get_data_from_db()

    # Search feature
    search_term = st.text_input('Enter search term')

    # Button to perform Image Scraping and OCR
    if st.button("Perform Image Scraping and OCR"):
        st.text("Performing Image Scraping and OCR...")
        perform_image_scraping_ocr()
        st.text("Image Scraping and OCR completed!")

        # Rerun the app after scraping
        df = get_data_from_db()
        st.experimental_rerun()

    search_results = df[df['ocr_result'].str.contains(search_term, na=False)]
    page_size = 5
    page_number = st.number_input(
        label="Page Number", min_value=1, max_value=len(search_results)//page_size+1, step=1)

    current_start = (page_number-1)*page_size
    current_end = page_number*page_size

    # Show OCR results, predictions, and image URLs in a single table
    for index, row in search_results.iloc[current_start:current_end].iterrows():
        # Determine row color based on prediction
        if row['prediction'] == 'bias':
            row_color = '#ffc0cb'
        else:
            row_color = '#fff'

        st.markdown(
        f"""
        <table style='background-color: #283e4a; color: black;'> <!-- Ubah color: #fff menjadi color: black -->
            <tr style='background-color: {row_color};'>
                <td style='width: 400px; text-align: center; vertical-align: middle;'><b>Job Description</b></td>
                <td style='text-align: center; vertical-align: middle;'><b>Identification Job</b></td>
                <td style='text-align: center; vertical-align: middle;'><b>Image URL</b></td>
            </tr>
            <tr style='background-color: {row_color};'>
                <td style='width: 400px; text-align: center; vertical-align: middle;'>{row['ocr_result']}</td>
                <td style='text-align: center; vertical-align: middle;'>{row['prediction']}</td>
                <td style='text-align: center; vertical-align: middle;'><a href="{row['image_url']}">link</a></td>
            </tr>
        </table>
        """,
        unsafe_allow_html=True)

    st.header("Text Classification Job Description to Bias or Non-Bias")
    text = st.text_area('Text to predict', 'Enter text here...')
    if st.button('Predict'):
        prediction = classify_text(text)
        st.write(prediction)

if __name__ == "__main__":
    main()
