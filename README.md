# Bangalore News RAG Chatbot

This project is a Django-based web application that implements a Retrieval Augmented Generation (RAG) pipeline to provide factual summaries and answers based on a dataset of Bangalore news. Users can query the system, and the application will retrieve relevant news chunks and use a Large Language Model (LLM) to generate informed responses.

## Features

* **Django Web Interface:** A user-friendly web interface built with Django for submitting queries and displaying answers.
* **RAG Pipeline:** Integrates a RAG workflow using:
    * **Data Loading:** Reads news articles from a CSV file (`kf_docmnt_export.csv`).
    * **Sentence Embeddings:** Utilizes `sentence-transformers` ('all-MiniLM-L6-v2') to convert text into numerical embeddings.
    * **FAISS Indexing:** Employs FAISS for efficient similarity search to retrieve relevant news chunks.
    * **LLM Integration:** Leverages OpenAI's API (`gpt-4o-mini` by default) to generate coherent answers based on the retrieved context and a user query.
* **Agent Personas:** Supports multiple AI agent personas (e.g., General News Reporter, Education & Campus News, Bangalore Weather Watch) with specialized system prompts and keyword filtering for more targeted responses.
* **Modular Design:** Separates the core RAG logic into a `rag_core` app for better organization.

## Getting Started

### Prerequisites

* Python 3.8+
* pip (Python package installer)
* An OpenAI API Key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/my-ai-showcase.git](https://github.com/your-username/my-ai-showcase.git)
    cd my-ai-showcase
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Prepare your data:**
    * Place your news data CSV file named `kf_docmnt_export.csv` in the root directory of the `my_ai_showcase` project (same level as `manage.py`).
    * Ensure the CSV has a column with the main text content, which is configured as `'DOC_DET'` in `rag_core/rag_pipeline.py`. If your column name is different, update the `text_column_name` variable in `rag_core/rag_pipeline.py`.

5.  **Set your OpenAI API Key:**
    * Open `my_ai_showcase/settings.py`.
    * Replace `'YOUR_OPENAI_API_KEY'` with your actual OpenAI API key:
        ```python
        OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', 'YOUR_OPENAI_API_KEY_HERE')
        ```
        It's highly recommended to set your OpenAI API key as an environment variable (e.g., `export OPENAI_API_KEY='sk-your-key'`) rather than hardcoding it directly in `settings.py` for production environments.

### Running the Application

1.  **Apply Django migrations:**
    ```bash
    python manage.py migrate
    ```

2.  **Run the Django development server:**
    ```bash
    python manage.py runserver
    ```

3.  **Access the application:**
    Open your web browser and go to `http://127.0.0.1:8000/`.

The first time you run the application and access the home page, the RAG pipeline components (embedding model and FAISS index) will be initialized and loaded. This might take some time depending on the size of your CSV file and your internet connection for downloading the `SentenceTransformer` model. The FAISS index will be saved to `bangalore_news_index.faiss` to speed up subsequent loads.

## Project Structure

* `my_ai_showcase/`: The main Django project directory.
    * `settings.py`: Django settings, including paths for CSV and FAISS index, and OpenAI API key.
    * `urls.py`: Main URL routing for the project.
* `my_ai_showcase/rag_core/`: Contains the core RAG pipeline logic.
    * `rag_pipeline.py`: Implements `RAGPipeline` class for data loading, embedding, FAISS indexing, retrieval, and LLM interaction.
* `my_ai_showcase/display_app/`: The Django application responsible for the web interface.
    * `views.py`: Handles web requests, calls the RAG pipeline, and manages agent personas.
    * `urls.py`: Defines URLs for the `display_app`.
    * `templates/home.html`: The HTML template for the main page.
    * `static/`: Contains static files like CSS.
* `kf_docmnt_export.csv`: Your dataset of Bangalore news articles (must be present in the root directory).
* `bangalore_news_index.faiss`: The FAISS index file, generated on first run.

## Contributing

Feel free to fork the repository, open issues, or submit pull requests.