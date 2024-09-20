# Automated-Text-Generation-from-PDF-Queries-Exam-Quiz

This Streamlit application allows users to upload a PDF, ask questions about the content, and generate exam questions (Multiple Choice Questions and Fill-in-the-Blanks) based on the uploaded PDF. It uses LangChain and Hugging Face embeddings to handle text queries and document splitting. The application also integrates ChatGroq for advanced language model processing.

## Features

- Upload PDF and retrieve content-based questions.
- Generate multiple-choice and fill-in-the-blank questions from the uploaded PDF.
- Chat interface to query the PDF for text-based answers.
- Interactive exam with answer validation.

## Requirements

The following Python packages are required to run the application:

- `streamlit`
- `os`
- `langchain_groq`
- `langchain_community.document_loaders`
- `langchain_community.vectorstores`
- `langchain.text_splitter`
- `langchain_core.prompts`
- `langchain.chains`
- `langchain_huggingface.embeddings`
- `dotenv`
- `random`
- `time`
- `warnings`

## Installation

1. Clone the repository to your local machine.

   ```bash
   git clone https://github.com/your-repository-name.git
   cd your-repository-name
   ```

2. Create a virtual environment and activate it:

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies using `pip`:

   ```bash
   pip install streamlit langchain_groq langchain_community langchain langchain_huggingface python-dotenv
   ```

4. Make sure you have the following environment variables set up in a `.env` file:

   ```
   GROQ_API_KEY=your_groq_api_key_here
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key_here
   ```

## Usage

1. Start the Streamlit app by running the following command:

   ```bash
   streamlit run app.py
   ```

2. In the sidebar, upload a PDF file. Once the file is uploaded, the app will process the PDF and split the document into smaller chunks using LangChain’s text splitter.

3. You can interact with two main sections:
   - **Home:** Query the content of the PDF via chat.
   - **Exam:** Generate multiple-choice and fill-in-the-blank questions based on the content of the PDF.

## Key Libraries

- **Streamlit**: For building the web interface.
- **LangChain**: Used for document loaders, splitting documents, creating prompts, and managing the chain for document retrieval.
- **HuggingFaceEmbeddings**: For generating embeddings from the uploaded PDF content.
- **FAISS**: Vector store used for efficient similarity searches.
- **ChatGroq**: Large language model for handling advanced natural language processing tasks.
- **dotenv**: For loading environment variables.

## Contributing

Feel free to submit pull requests to add new features or improve existing functionality. Before submitting, ensure that you run the following tests:

```bash
pytest
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
```

### Instructions to Set Up:
1. **Make sure you have the `.env` file** with the API keys for Hugging Face and Groq.
2. **Install dependencies** using `pip`.
3. **Run the Streamlit application** as described in the usage section.

This `README.md` should guide you through setting up the project and getting it running smoothly!
