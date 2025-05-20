import os
import pandas as pd # Although pandas is mostly used in rag_pipeline, it's good to keep if direct data interaction were here
from django.shortcuts import render
from django.conf import settings # To access settings like API key and file paths
from rag_core.rag_pipeline import RAGPipeline # Import your RAG pipeline class

# --- Global variable to hold RAGPipeline instance ---
# This ensures the heavy RAG components (model, index) are loaded only once
_rag_pipeline = None

def load_rag_components_once():
    """
    Loads and initializes the RAGPipeline components (model, FAISS index, data)
    only once when the server starts or the first request comes in.
    """
    global _rag_pipeline
    if _rag_pipeline is None:
        csv_path = settings.CSV_FILE_PATH
        index_path = settings.FAISS_INDEX_PATH
        _rag_pipeline = RAGPipeline(csv_path, index_path)
        print("RAG Pipeline components loaded successfully!") # For debugging
    return _rag_pipeline

# --- Define Agent Configurations ---
# This dictionary holds all the information for each AI agent,
# including display name, system prompt for the LLM, and keywords for context filtering.
AGENT_CONFIGS = {
    "general_news": {
        "display_name": "General News Reporter",
        "system_prompt": (
            "You are a helpful AI assistant providing concise and factual summaries based on "
            "Bangalore news. Answer the question based ONLY on the provided context. "
            "If the context doesn't contain the answer, state that the information is not "
            "available in the provided documents. Do not make up information."
        ),
        "keywords_for_filtering": [] # No specific filtering for general news
    },
    "education_news": {
        "display_name": "Education & Campus News",
        "system_prompt": (
            "You are an 'Education & Campus News' assistant for Bangalore. "
            "Based ONLY on the provided news context, answer questions about "
            "schools, colleges, university announcements, educational policy changes, "
            "exams, or significant campus events. If the news context doesn't have "
            "this information, state so."
        ),
        "keywords_for_filtering": [
            "school", "college", "university", "student", "exam", "admission", "syllabus",
            "education policy", "campus", "sslc", "puc", "results", "academic", "board",
            "institution", "degree", "semester", "curriculum", "faculty", "research"
        ]
    },
    "bangalore_weather": {
        "display_name": "Bangalore Weather Watch (News-Based)",
        "system_prompt": "You are 'Bangalore Weather Watch.' Report ONLY on significant weather events or official advisories for Bangalore found in the provided news context. Do not provide live forecasts or general weather knowledge. If no relevant news is in the context, state that.",
        "keywords_for_filtering": [
            "weather", "rain", "rainfall", "temperature", "heatwave", "monsoon", "imd",
            "forecast", "climate", "humidity", "cyclone", "storm", "flood", "dry spell"
        ]
    },
    "local_safety": {
        "display_name": "Local Safety Reporter",
        "system_prompt": (
            "You are a 'Local Safety Reporter' for Bangalore. Your role is to provide factual "
            "information based ONLY on the provided news context about reported crime incidents, "
            "accidents, and public safety announcements. Be objective and stick strictly to the "
            "details in the reports. If the context does not contain relevant safety "
            "information for the query, state that."
        ),
        "keywords_for_filtering": [
            "police", "arrest", "crime", "theft", "accident", "safety", "alert", "fir",
            "assault", "scam", "fraud", "emergency", "complaint", "victim", "investigation",
            "security", "patrol", "incident"
        ]
    },
    "community_corner": {
        "display_name": "Community Corner",
        "system_prompt": (
            "You are 'Community Corner,' an AI assistant highlighting local community news and events in Bangalore "
            "based ONLY on the provided news context. Report on events, festivals, initiatives, and community activities. "
            "If the context lacks this information, state that."
        ),
        "keywords_for_filtering": [
            "community", "event", "festival", "celebration", "initiative", "local", "group",
            "organization", "volunteer", "fair", "gathering", "activity", "workshop", "public"
        ]
    },
    "public_transport": {
        "display_name": "Public Transport Pulse",
        "system_prompt": (
            "You are 'Public Transport Pulse,' an AI specializing in Bangalore's public transportation news. "
            "Using ONLY the provided news context, answer questions about Namma Metro or BMTC buses, "
            "including reported service updates, new routes, planned expansions, significant disruptions, "
            "or fare changes. If the context has no relevant public transport news for the query, "
            "clearly state that."
        ),
        "keywords_for_filtering": [
            "metro", "namma metro", "bmrcl", "bmtc", "bus", "public transport", "commute",
            "commuters", "fare", "route", "station", "transportation", "service", "feeder service",
            "last mile", "depot", "corridor", "line"
        ]
    },
    # Add other agent configurations here if needed
}

def get_specialized_context_for_agent(query_text, agent_id, all_chunks, top_n=3):
    """
    Filters retrieved chunks based on keywords relevant to the selected agent.
    This helps provide more targeted context to the LLM.
    """
    agent_config = AGENT_CONFIGS.get(agent_id)
    # If agent_id not found or no specific keywords, return a default number of chunks
    if not agent_config or not agent_config.get("keywords_for_filtering"):
        return all_chunks[:top_n]

    keywords = [keyword.lower() for keyword in agent_config["keywords_for_filtering"]]
    filtered_chunks = []

    for chunk in all_chunks:
        text_lower = chunk['chunk_text'].lower()
        if any(keyword in text_lower for keyword in keywords):
            filtered_chunks.append(chunk)

    # Return up to top_n of the filtered chunks
    return filtered_chunks[:top_n]

def home_view(request):
    """
    Handles the main web page for the RAG application.
    Processes user queries and returns answers from the selected AI agent.
    """
    rag_pipeline = load_rag_components_once() # Ensure RAG components are loaded
    answer = None
    selected_agent_id = request.POST.get('agent_persona', 'general_news') # Get selected agent from form
    user_query = request.POST.get('user_query', '').strip() # Get user query from form, strip whitespace

    if request.method == 'POST' and user_query: # Only process if it's a POST request and query is not empty
        print(f"User Query: '{user_query}' with Agent: '{selected_agent_id}'") # Debugging
        
        # 1. Retrieve initial broad relevant chunks from the FAISS index
        retrieved_chunks = rag_pipeline.retrieve_relevant_chunks(user_query, top_k=7) # Get more chunks initially

        # 2. Apply agent-specific context filtering
        if selected_agent_id != 'general_news':
            specialized_context = get_specialized_context_for_agent(
                user_query, selected_agent_id, retrieved_chunks, top_n=3 # Pass filtered chunks to LLM
            )
            agent_config = AGENT_CONFIGS.get(selected_agent_id)
            system_prompt = agent_config.get("system_prompt")
        else:
            # For general news, use all retrieved chunks without further filtering
            specialized_context = retrieved_chunks[:3] # Default to top 3 for general
            system_prompt = AGENT_CONFIGS['general_news']['system_prompt']

        # Fallback if no context was found after filtering
        if not specialized_context:
            answer = "I'm sorry, I couldn't find relevant information in the news for your query, even with the selected agent's focus."
        else:
            # 3. Generate answer using the LLM with the specialized context and system prompt
            answer = rag_pipeline.generate_answer_with_llm(user_query, specialized_context, system_prompt)
        
        print(f"Generated Answer: {answer[:100]}...") # Debugging

    # Render the home.html template, passing data to it
    return render(request, 'home.html', {
        'answer': answer,
        'agent_configs': AGENT_CONFIGS,
        'selected_agent': selected_agent_id,
        'user_query': user_query # To re-populate the query box after submission
    })

# Note: display_app/admin.py, display_app/apps.py, display_app/models.py, display_app/tests.py can be left empty
# or deleted if not used for their respective functionalities.