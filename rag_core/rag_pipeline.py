import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI
from django.conf import settings # To access Django settings (like API key)

class RAGPipeline:
    def __init__(self, csv_file_path, index_file_path):
        self.csv_file_path = csv_file_path
        self.index_file_path = index_file_path
        self.data = pd.DataFrame() # Initialize as empty DataFrame
        self.model = None
        self.index = None

        print("Initializing RAGPipeline components...")
        self._initialize_components()
        if self.data.empty or self.model is None or self.index is None:
            print("RAGPipeline initialization FAILED. One or more components are not loaded.")
        else:
            print("RAGPipeline initialized SUCCESSFULLY.")


    def _initialize_components(self):
        """Initializes data, embedding model, and FAISS index."""
        # 1. Load Data
        self.data = self._load_data()
        if self.data.empty:
            print("Status: Data loading FAILED or returned empty DataFrame.")
            return # Stop initialization if data is not loaded

        # 2. Load Embedding Model
        self.model = self._load_embedding_model()
        if self.model is None:
            print("Status: Embedding model loading FAILED.")
            return # Stop initialization if model is not loaded

        # 3. Load or Create FAISS Index
        self.index = self._load_faiss_index()
        if self.index is None:
            print("Status: FAISS index loading/creation FAILED.")
            return # Stop initialization if index is not loaded/created


    def _load_embedding_model(self):
        """Loads the SentenceTransformer model."""
        try:
            print("Attempting to load SentenceTransformer model 'all-MiniLM-L6-v2'...")
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("SentenceTransformer model loaded successfully.")
            return model
        except Exception as e:
            print(f"ERROR: Failed to load SentenceTransformer model: {e}")
            print("Please check your internet connection or if the model files are corrupted.")
            return None

    def _load_data(self):
        """Loads data from the CSV file."""
        try:
            print(f"Attempting to load data from CSV: {self.csv_file_path}")
            df = pd.read_csv(self.csv_file_path)

            # --- CRITICAL FIX START ---
            # Define the actual column that holds your main text content
            # Based on your provided columns, 'DOC_DET' or 'DOC_DESC' are likely candidates.
            # You MUST verify this in your actual CSV file.
            text_column_name = 'DOC_DET' # <--- CONFIRM THIS IS THE CORRECT COLUMN IN YOUR CSV
            # If it's 'DOC_DESC', change it to 'DOC_DESC'. If it's another, use that name.

            if text_column_name not in df.columns:
                print(f"ERROR: Specified text column '{text_column_name}' not found in CSV. Available columns: {df.columns.tolist()}")
                return pd.DataFrame()

            # Rename the actual text column to 'content' for internal consistency
            df = df.rename(columns={text_column_name: 'content'})
            # --- CRITICAL FIX END ---

            # Optional: Drop rows where 'content' might be empty after renaming
            df = df.dropna(subset=['content']).reset_index(drop=True)

            if 'content' not in df.columns: # This check is now mostly for double-checking after rename
                print(f"ERROR: After renaming, 'content' column is still not found. This should not happen if text_column_name was correct.")
                return pd.DataFrame()

            print(f"Data loaded successfully. Rows: {len(df)}")
            return df
        except FileNotFoundError:
            print(f"ERROR: CSV file not found at {self.csv_file_path}")
            return pd.DataFrame()
        except pd.errors.EmptyDataError:
            print(f"ERROR: CSV file at {self.csv_file_path} is empty.")
            return pd.DataFrame()
        except Exception as e:
            print(f"ERROR: An unexpected error occurred loading CSV file: {e}")
            return pd.DataFrame()

    def _create_faiss_index(self, embeddings):
        """Creates a new FAISS index and saves it to disk."""
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32')) # Ensure embeddings are float32
        faiss.write_index(index, self.index_file_path)
        print(f"FAISS index created and saved to {self.index_file_path}")
        return index

    def _load_faiss_index(self):
        """Loads an existing FAISS index or creates a new one if it doesn't exist."""
        if os.path.exists(self.index_file_path):
            try:
                print(f"Attempting to load FAISS index from {self.index_file_path}...")
                index = faiss.read_index(self.index_file_path)
                print("FAISS index loaded successfully.")
                return index
            except Exception as e:
                print(f"ERROR: Failed to load FAISS index from {self.index_file_path}: {e}")
                print("Attempting to recreate the index due to loading error...")
                if not self.data.empty and self.model is not None:
                    embeddings = self.model.encode(self.data['content'].tolist())
                    return self._create_faiss_index(embeddings.astype('float32'))
                else:
                    print("Cannot recreate index: Data or model not loaded/available.")
                    return None
        else:
            print("FAISS index not found. Attempting to create a new one...")
            if not self.data.empty and self.model is not None:
                embeddings = self.model.encode(self.data['content'].tolist())
                return self._create_faiss_index(embeddings.astype('float32'))
            else:
                print("Cannot create index: Data or model not loaded/available.")
                return None

    def retrieve_relevant_chunks(self, query, top_k=5):
        """Retrieves the most relevant text chunks from the FAISS index."""
        if self.index is None or self.data.empty or self.model is None:
            print("Retrieval Warning: RAG Pipeline not fully initialized (index/data/model is None).")
            return []
        try:
            query_embedding = self.model.encode([query]).astype('float32')
            # D = distances, I = indices
            D, I = self.index.search(query_embedding, top_k)
            relevant_chunks = []
            for i in range(len(I[0])):
                index = I[0][i]
                # Check if the retrieved index is valid and within the bounds of the DataFrame
                if 0 <= index < len(self.data):
                    relevant_chunks.append({'chunk_text': str(self.data['content'].iloc[index])}) # Ensure content is string
                else:
                    print(f"Retrieval Warning: Index {index} from FAISS is out of data bounds (size {len(self.data)}).")
            print(f"Retrieved {len(relevant_chunks)} chunks for query: '{query}'.")
            return relevant_chunks
        except Exception as e:
            print(f"ERROR: An error occurred during chunk retrieval: {e}")
            return []

    def generate_answer_with_llm(self, query, retrieved_chunks_info, system_message_content, llm_model="gpt-4o-mini"):
        """Generates an answer using the LLM based on the query and retrieved context."""
        openai_api_key = settings.OPENAI_API_KEY
        if not openai_api_key or openai_api_key == 'YOUR_OPENAI_API_KEY_HERE':
            print("ERROR: OpenAI API key is missing or not set.")
            return "Error: OpenAI API key is not configured. Please set OPENAI_API_KEY in settings.py or as an environment variable."

        context_str = "\n\n".join([chunk['chunk_text'] for chunk in retrieved_chunks_info]) if retrieved_chunks_info else "No relevant context found."
        print(f"Sending to LLM (Model: {llm_model}):\nSystem Prompt: {system_message_content[:50]}...\nContext: {context_str[:200]}...\nQuery: {query}") # Debugging LLM input

        try:
            client = OpenAI(api_key=openai_api_key)
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_message_content},
                    {"role": "user", "content": f"Based on the following context:\n\nContext:\n{context_str}\n\nQuestion: {query}\n\nAnswer:"}
                ],
                model=llm_model,
                max_tokens=300,
                temperature=0.3
            )
            answer = chat_completion.choices[0].message.content.strip()
            print("LLM Answer generated.")
            return answer
        except Exception as e:
            print(f"ERROR: Failed to generate answer with LLM: {e}")
            return f"An error occurred while generating the answer: {e}. Please check your OpenAI API key and internet connection."