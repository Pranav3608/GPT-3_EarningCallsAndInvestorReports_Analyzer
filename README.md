# OpenAI-GPT3_InvestorReports_Analyzer
Streamlit-based web app using OpenAI's GPT-3 model to summarize financial transcripts, extract key points and discover actionable insights. Utilizes ML, NLP and Genrative AI techniques for efficient content analysis. The application also utilises Micorsoft Azure blob storage for efficient data storage and retrieval. 

**Installation**

To run this project locally, you need to install the following dependencies:

- **streamlit**
- **openai**
- **numpy**
- **base64**
- **uuid**
- **azure.storage.blob**
- **nltk**
- **scikit-learn**
- **networkx**

You can install these dependencies by running the following command:

bashCopy code

pip install streamlit openai numpy base64 uuid azure-storage-blob nltk scikit-learn networkx 

**Usage**

1. Upload your transcript file by clicking on the file upload button.
1. Once the file is uploaded, the app will display a preview of the first three lines of the text.
1. The file will be processed and summarized using OpenAI's text summarization model.
1. The full summary and key points will be displayed on the app.
1. You can also download the summary as a text file by entering a file name and clicking the "Download Summary" button.

**Configuration**

Before running the app, make sure to set up the following configuration variables:

- **openai.api\_key**: Your OpenAI API key.
- **storage\_connection\_string**: Connection string for your Azure Storage account.
- **container\_name**: Name of the container in Azure Blob Storage where the transcript files will be uploaded.
- **output\_container\_name**: Name of the container in Azure Blob Storage where the output summary files will be stored.

**Acknowledgements**

This project utilizes OpenAI's text summarization API and Azure Blob Storage for file storage. It also uses various Python libraries, including Streamlit, NumPy, NLTK, scikit-learn, and NetworkX.

**Limitations**

Please note that the summarization quality may vary depending on the complexity and length of the transcript. The app is currently configured to generate a concise summary using the first 10 chunks of the transcript.

