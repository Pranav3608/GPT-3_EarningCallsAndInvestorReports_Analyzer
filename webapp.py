import streamlit as st
import openai
import numpy as np
import base64
import uuid
from azure.storage.blob import BlobServiceClient
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

# Streamlit setup
st.set_page_config(layout="wide")
st.title('OpenAI Transcript Summariser')
st.text('This is a web app to allow Summarization of Transcripts')
upload_file = st.file_uploader('Upload your transcript file')

# OpenAI setup
openai.api_type = "azure"
openai.api_base = "Azure_API_endpoint"
openai.api_version = "2022-12-01"
openai.api_key = 'Your_OpenAI_API_Key'

# Azure Storage setup
storage_connection_string = "Your_Azure_Storageconnection_String"
container_name = "transcriptfile"
output_container_name = "output-summary-files"  # New container for output summary
blob_service_client = BlobServiceClient.from_connection_string(storage_connection_string)

# Create the output container if it doesn't exist
if output_container_name not in [container.name for container in blob_service_client.list_containers()]:
    blob_service_client.create_container(output_container_name)


def generate_key_points(text):
    # Tokenize the text into sentences
    sentences = sent_tokenize(text)

    # Remove special characters and numbers from sentences
    clean_sentences = [re.sub(r"[^a-zA-Z\s]", "", sentence) for sentence in sentences]

    # Create a CountVectorizer to convert sentences into a matrix of token counts
    vectorizer = CountVectorizer(stop_words='english')
    sentence_vectors = vectorizer.fit_transform(clean_sentences)

    # Calculate sentence similarities using cosine similarity
    similarity_matrix = cosine_similarity(sentence_vectors, sentence_vectors)

    # Convert similarity matrix to graph representation
    graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(graph)

    # Sort sentences based on scores
    ranked_sentences = sorted(((scores[i], sentence) for i, sentence in enumerate(sentences)), reverse=True)

    # Select top N key points
    num_key_points = 10  # Specify the desired number of key points here
    key_points = [ranked_sentence[1] for _, ranked_sentence in enumerate(ranked_sentences) if _ < num_key_points]

    return key_points


if upload_file is not None:
    # Read the file as text
    text = upload_file.read().decode("utf-8")

    # Display the first three lines of the text as a preview
    lines = text.split('\n')
    if len(lines) > 3:
        preview_text = '\n'.join(lines[:3]) + " ..."
    else:
        preview_text = '\n'.join(lines)
    st.text('File Preview:')
    st.text(preview_text)

    # Get the original file name
    file_name = upload_file.name

    # Generate a unique name for the uploaded file
    unique_file_name = f"{str(uuid.uuid4())}_{file_name}"

    # Upload file to Azure Blob Storage
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=unique_file_name)
    blob_client.upload_blob(text)
    st.success('File uploaded successfully!')

    # Rest of the code for summarization...
    # Placeholder for the status message
    status_placeholder = st.empty()

    summary_responses = []

    # Display the status message
    status_placeholder.text('Generating summary... Please wait.')

    # Retrieve the uploaded file from Azure Blob Storage
    uploaded_blob_client = blob_service_client.get_blob_client(container=container_name, blob=unique_file_name)
    uploaded_text = uploaded_blob_client.download_blob().readall().decode("utf-8")

    # Chunking up the data
    words = uploaded_text.split(" ")
    chunks = np.array_split(words, 10)

    for chunk in chunks:
        sentences = ' '.join(list(chunk))
        prompt = f"Generate a concise summary of the following text:\n\n{sentences}\n\n"

        response = openai.Completion.create(
            engine="text-summary-poc",
            prompt=prompt,
            temperature=0.2,
            max_tokens=100,
            top_p=1,
            frequency_penalty=0.5,
            presence_penalty=1.5
        )
        response_text = response.choices[0].text.strip()
        summary_responses.append(response_text)

    full_summary = " ".join(summary_responses)

    # Check if the last sentence is incomplete and truncate if necessary
    last_sentence = full_summary.split('.')[-1].strip()
    if last_sentence and last_sentence[-1] != '.':
        full_summary = full_summary[:-(len(last_sentence))].strip()

    # Generate key points from the summary
    key_points = generate_key_points(full_summary)

    # Add spacing between summary and key points
    st.markdown("<br>", unsafe_allow_html=True)

    # Display the summary
    st.subheader('Full Summary:')
    st.markdown(f"<pre style='white-space: pre-wrap;'>{full_summary}</pre>", unsafe_allow_html=True)

    # Add spacing between summary and key points
    st.markdown("<br>", unsafe_allow_html=True)

    # Display the key points
    st.subheader('Key Points:')
    key_points_text = "\n".join([f"<li>{point}</li>" for point in key_points])
    key_points_html = f"<ul style='font-family: Arial, sans-serif;'>{key_points_text}</ul>"
    st.markdown(key_points_html, unsafe_allow_html=True)

    # Add spacing between key points and word count
    st.markdown("<br>", unsafe_allow_html=True)

    # Count words in input text file
    input_word_count = len(words)

    # Count words in summary
    summary_words = full_summary.split(" ")
    summary_word_count = len(summary_words)

    # Display word count boxes in parallel
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Number of words in input text file:\n{input_word_count}")
    with col2:
        st.info(f"Number of words in the summary:\n{summary_word_count}")

    # User input for the download file name
    input_file_name_without_extension = file_name.split(".")[0]
    download_file_name = st.text_input("Enter file name for download",
                                       value=f"{input_file_name_without_extension}_summary.txt")

    if st.button("Download Summary"):
        if download_file_name == "":
            st.warning("Please enter a valid filename for download.")
        else:
            # Generate download link with custom file name
            def get_download_link(text, filename):
                b64 = base64.b64encode(text.encode()).decode()
                href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Summary</a>'
                return href

            # Provide a download link for the summary
            download_link = get_download_link(full_summary, download_file_name)
            st.markdown(download_link, unsafe_allow_html=True)
            st.success("Summary downloaded successfully.")

            # Upload the summary to Azure Blob Storage
            output_blob_client = blob_service_client.get_blob_client(container=output_container_name,
                                                                    blob=download_file_name)
            output_blob_client.upload_blob(full_summary)

    # Remove the uploaded file from Azure Blob Storage
    uploaded_blob_client.delete_blob()
