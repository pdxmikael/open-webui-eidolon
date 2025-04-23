import os
import json
import random
import time
import argparse
import sqlite3 # Added for querying webui.db
from typing import List, Dict, Optional, Tuple
# --- Add heapq for efficient top-N ---
import heapq
import gc # Import garbage collector module
import datetime # Import datetime for timing

# --- Dependency Imports ---
try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    print("Error: chromadb library not found. Please install with 'pip install chromadb'")
    exit(1)

try:
    # Using sentence-transformers embedding function requires the library
    import sentence_transformers
except ImportError:
    print("Error: sentence-transformers library not found. Please install with 'pip install sentence-transformers'")
    exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("Warning: python-dotenv not found. Cannot load .env file. Ensure environment variables are set.")
    # Continue execution, assuming environment variables are set manually

try:
    import requests
except ImportError:
    print("Error: requests library not found. Please install with 'pip install requests'")
    exit(1)

# --- NEW: Import kneed ---
try:
    from kneed import KneeLocator
except ImportError:
    print("Error: kneed library not found. Please install with 'pip install kneed'")
    exit(1)
# --- End NEW ---

# --- LLM Client Imports ---
# Store imported clients to avoid repeated imports
_llm_clients = {}

def _import_llm_client(provider):
    global _llm_clients
    if provider in _llm_clients:
        return True

    try:
        if provider == "ollama":
            import ollama
            _llm_clients['ollama'] = ollama
        elif provider == "openai":
            import openai
            _llm_clients['openai'] = openai
        elif provider == "gemini":
            import google.generativeai as genai
            _llm_clients['gemini'] = genai
        else:
            print(f"Error: Unsupported LLM_PROVIDER '{provider}'. Use 'ollama', 'openai', or 'gemini'.")
            return False
        return True
    except ImportError as e:
        print(f"Error: Failed to import LLM library for provider '{provider}'.")
        # Suggest correct installation command based on provider
        install_command = "pip install "
        if provider == "ollama":
            install_command += "ollama"
        elif provider == "openai":
            install_command += "openai"
        elif provider == "gemini":
            install_command += "google-generativeai"
        else:
            install_command = f"(Unknown install command for provider: {provider})" # Should not happen

        print(f"Make sure '{provider}' library is installed (e.g., '{install_command}')")
        print(f"Import error: {e}")
        return False

# --- Load Environment Variables ---
load_dotenv()

# --- Helper function to read and clean env vars ---
def get_clean_env_var(key, default=None):
    value = os.getenv(key, default)
    if value is None:
        return None
    cleaned_value = value.split('#')[0].strip().strip('\'"')
    return cleaned_value

# --- Configuration (Read from Env for Defaults) ---
DEFAULT_VECTOR_DB_PATH = get_clean_env_var("VECTOR_DB_PATH")
# DEFAULT_COLLECTION_NAME is no longer needed as a primary config
DEFAULT_EMBEDDING_MODEL_NAME = get_clean_env_var("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
DEFAULT_LLM_PROVIDER = get_clean_env_var("LLM_PROVIDER", "ollama").lower()
DEFAULT_OLLAMA_BASE_URL = get_clean_env_var("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_OPENAI_API_KEY = get_clean_env_var("OPENAI_API_KEY")
DEFAULT_OPENAI_API_BASE = get_clean_env_var("OPENAI_API_BASE", "https://api.openai.com/v1")
DEFAULT_GEMINI_API_KEY = get_clean_env_var("GEMINI_API_KEY")
DEFAULT_LLM_MODEL = get_clean_env_var("LLM_MODEL", "llama3") # Adjust default based on provider?

DEFAULT_SEMANTIC_SEARCH_RESULTS_STR = get_clean_env_var("SEMANTIC_SEARCH_RESULTS", "5")
DEFAULT_MAX_RETRIES_STR = get_clean_env_var("MAX_RETRIES", "3")
RETRY_DELAY = 5 # seconds

# --- NEW: Default path for webui.db and tag name ---
DEFAULT_WEBUI_DB_PATH = get_clean_env_var("WEBUI_DB_PATH", "backend/data/webui.db")
DEFAULT_TAG_NAME = get_clean_env_var("TAG_NAME", "CrusaderKingsIII") # Default to the name user mentioned

# Convert numeric defaults safely
try:
    DEFAULT_SEMANTIC_SEARCH_RESULTS = int(DEFAULT_SEMANTIC_SEARCH_RESULTS_STR)
except (ValueError, TypeError):
    print(f"Warning: Invalid value '{DEFAULT_SEMANTIC_SEARCH_RESULTS_STR}' for SEMANTIC_SEARCH_RESULTS in .env. Using default 5.")
    DEFAULT_SEMANTIC_SEARCH_RESULTS = 5
try:
    DEFAULT_MAX_RETRIES = int(DEFAULT_MAX_RETRIES_STR)
except (ValueError, TypeError):
    print(f"Warning: Invalid value '{DEFAULT_MAX_RETRIES_STR}' for MAX_RETRIES in .env. Using default 3.")
    DEFAULT_MAX_RETRIES = 3

# --- NEW: Re-initialization and Sampling Control ---
ENABLE_CLIENT_REINIT = True # Set to False to disable periodic re-initialization
CLIENT_REINIT_INTERVAL = 100 # Re-initialize client every N collections if enabled
MAX_COLLECTIONS = 100 # Max collections to search (0 = search all). Randomly samples if > 0 and < total valid collections. # Changed to 100
MAX_PIPELINE_ATTEMPTS = 5 # Max attempts for the entire generation pipeline (sampling -> LLM calls)

# --- Prompts (Customize these heavily!) ---
CONTEXT_VALIDATION_SYSTEM_PROMPT = "You are an expert Crusader Kings III game designer."
CONTEXT_VALIDATION_PROMPT_TEMPLATE = """Analyze the following Crusader Kings III script excerpts. Determine if the provided script context contains meaningful information about a game mechanic or potentially serves as a valuable example for designers.

Respond with ONLY one of the following assessments:
- "Sufficient"
- "Insufficient - Likely Missing Definition: [Specify missing element, e.g., 'some_flag']"
- "Insufficient - Context Fragmented"
- "Insufficient - Other Reason: [Briefly explain]"

Script Context:
---
{context}
---

Assessment:"""

INTERPRETATION_SYSTEM_PROMPT = "You are an expert Crusader Kings III game designer. Your task is to interpret game script and explain the game mechanics they result in in clear, natural language."
INTERPRETATION_PROMPT_TEMPLATE = """Based *only* on the following Crusader Kings III script excerpts, explain the game mechanic, event, interaction, or definition they represent in clear, natural language suitable for someone learning about the game. Focus on what the player would observe or experience.

Script Context:
---
{context}
---

Natural Language Explanation:"""

EXPLANATION_REVIEW_SYSTEM_PROMPT = "You are a meticulous editor reviewing explanations of Crusader Kings III mechanics based on game scripts."
EXPLANATION_REVIEW_PROMPT_TEMPLATE = """Review the following natural language explanation against the original script context it was derived from.

Check for:
1.  **Accuracy:** Does the explanation correctly reflect the logic and details in the script?
2.  **Clarity:** Is the explanation easy to understand?
3.  **Completeness:** Does it cover the main points implied by the script context?
4.  **Conciseness:** Is it free of unnecessary jargon or speculation beyond the script?

Respond ONLY with one of the following assessments, optionally followed by a brief reason if not 'Clear and Accurate':
- "Clear and Accurate"
- "Inaccurate: [Brief reason]"
- "Unclear: [Brief reason]"
- "Incomplete: [Brief reason]"

Script Context:
---
{script_context}
---

Natural Language Explanation:
---
{explanation}
---

Review Assessment:"""

QA_GENERATION_SYSTEM_PROMPT = "You are tasked with creating high-quality training data for a Crusader Kings III expert AI."
QA_GENERATION_PROMPT_TEMPLATE = """Based *only* on the provided natural language explanation of a Crusader Kings III mechanic, generate one relevant question a player might ask and a comprehensive, accurate answer derived strictly from the explanation.

**Important:** The answer should directly address the question without using introductory phrases like "Based on the explanation..." or "According to the text...".

Format the output as a JSON object with keys "question" and "answer".

Explanation:
---
{explanation}
---

JSON Output:"""

QA_REVIEW_SYSTEM_PROMPT = "You are evaluating the quality of generated Question/Answer pairs for training a Crusader Kings III expert AI."
QA_REVIEW_PROMPT_TEMPLATE = """Evaluate the following Question/Answer pair based on the provided explanation.

Check:
1.  **Question Relevance:** Is the question natural, specific, and relevant to the explanation and typical CK3 gameplay?
2.  **Answer Accuracy:** Does the answer accurately reflect the information *only* within the explanation?
3.  **Answer Completeness:** Does the answer fully address the question based on the explanation?
4.  **Overall Quality:** Is the pair suitable for training an expert system?

Respond ONLY with one of the following assessments:
- "High Quality"
- "Low Quality - Question Irrelevant/Vague"
- "Low Quality - Answer Inaccurate/Incomplete"
- "Low Quality - Answer Goes Beyond Explanation"
- "Low Quality - Other: [Brief reason]"

Explanation:
---
{explanation}
---

Question:
{question}

Answer:
{answer}
---

Quality Assessment:"""

# --- Argument Parser ---
# Use cleaned env vars as defaults
parser = argparse.ArgumentParser(description="Generate CK3 Q&A pairs using documents tagged in webui.db.")
parser.add_argument("--db-path", default=DEFAULT_VECTOR_DB_PATH, help="Path to the ChromaDB database directory.")
# Removed --collection argument
parser.add_argument("--webui-db-path", default=DEFAULT_WEBUI_DB_PATH, help="Path to the webui.db SQLite file.") # Added
parser.add_argument("--tag-name", default=DEFAULT_TAG_NAME, help="The tag name in webui.db identifying the relevant documents.") # Added
parser.add_argument("--embed-model", default=DEFAULT_EMBEDDING_MODEL_NAME, help="Name of the sentence-transformer embedding model.")
parser.add_argument("--llm-provider", default=DEFAULT_LLM_PROVIDER, choices=['ollama', 'openai', 'gemini'], help="LLM provider to use.")
parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="The specific LLM model name to use.")
parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_BASE_URL, help="Base URL for Ollama API.")
parser.add_argument("--openai-key", default=DEFAULT_OPENAI_API_KEY, help="OpenAI API Key.")
parser.add_argument("--openai-base", default=DEFAULT_OPENAI_API_BASE, help="OpenAI API Base URL.")
parser.add_argument("--gemini-key", default=DEFAULT_GEMINI_API_KEY, help="Google Gemini API Key.")
# --- Updated Chunk Count Arguments ---
parser.add_argument("--num-results-per-coll", type=int, default=3, help="Max number of chunks to retrieve *per collection* during search.") # Changed to 3
parser.add_argument("--max-context-docs", type=int, default=50, help="Max number of unique document chunks to aim for in the final context (total).") # Default 50
# --- REMOVED: Distance Threshold Argument ---
# parser.add_argument("--max-distance-threshold", type=float, default=1.0, help="Maximum distance (lower is more similar) for chunks to be included in the context.")
# --- End Updated Chunk Count Arguments ---
parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Max retries for LLM calls.")
parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging.")

args = parser.parse_args() # Let argparse handle defaults and overrides

# --- Validate Configuration ---
# Use the final resolved values from the args object for validation
if not args.db_path:
    print("Error: ChromaDB path must be provided via .env file or --db-path argument.")
    exit(1)
if not args.webui_db_path: # Added validation
    print("Error: webui.db path must be provided via .env file or --webui-db-path argument.")
    exit(1)
if not args.tag_name: # Added validation
    print("Error: Tag name must be provided via .env file or --tag-name argument.")
    exit(1)
if args.llm_provider == "openai" and not args.openai_key:
    print("Error: OpenAI API key must be provided for the 'openai' provider.")
    exit(1)
if args.llm_provider == "gemini" and not args.gemini_key:
    print("Error: Gemini API key must be provided for the 'gemini' provider via --gemini-key or GEMINI_API_KEY env var.")
    exit(1)

# Attempt to import the required LLM client library early
if not _import_llm_client(args.llm_provider):
    exit(1)

# --- Logging ---
def log_verbose(message):
    if args.verbose:
        print(f"VERBOSE: {message}")

# --- LLM Interaction Function ---
def call_llm(prompt: str, system_message: Optional[str] = None) -> Optional[str]:
    """Calls the configured LLM with retry logic."""
    log_verbose(f"Calling LLM (Provider: {args.llm_provider}, Model: {args.llm_model})")
    # log_verbose(f"System Message: {system_message}") # System message is now part of the printed prompt below

    # Ensure client library is loaded
    if not _import_llm_client(args.llm_provider):
         return None # Should have exited earlier, but double-check

    messages = []
    # Gemini handles system prompts differently or not at all in some APIs/models
    # We'll prepend it to the user prompt for Gemini for simplicity here.
    effective_prompt = prompt
    full_prompt_display = "" # For printing

    if args.llm_provider != "gemini" and system_message:
        messages.append({"role": "system", "content": system_message})
        full_prompt_display = f"System: {system_message}\n\nUser: {prompt}"
    elif args.llm_provider == "gemini" and system_message:
        effective_prompt = f"{system_message}\n\n---\n\n{prompt}"
        log_verbose("Prepending system message to user prompt for Gemini.")
        full_prompt_display = f"(System prepended for Gemini)\n\n{effective_prompt}"
    else:
        # No system message
        full_prompt_display = f"User: {prompt}"


    # Gemini uses a different structure for messages (list of content, not role/content dicts)
    if args.llm_provider != "gemini":
        messages.append({"role": "user", "content": effective_prompt}) # Use effective_prompt here too

    # --- Echo Prompt ---
    print("\n--- LLM Prompt ---")
    print(full_prompt_display)
    print("--- End LLM Prompt ---")
    # --- End Echo Prompt ---


    for attempt in range(args.max_retries): # Use resolved value from args
        try:
            content = None # Initialize content to None for this attempt
            if args.llm_provider == "ollama":
                # ...existing ollama code...
                ollama_client = _llm_clients['ollama']
                client = ollama_client.Client(host=args.ollama_url)
                response = client.chat(
                    model=args.llm_model,
                    messages=messages
                )
                content = response['message']['content']
                # ...existing ollama code...

            elif args.llm_provider == "openai":
                # ...existing openai code...
                openai_client = _llm_clients['openai']
                client = openai_client.OpenAI(api_key=args.openai_key, base_url=args.openai_base)
                response = client.chat.completions.create(
                    model=args.llm_model,
                    messages=messages
                )
                content = response.choices[0].message.content
                # ...existing openai code...

            elif args.llm_provider == "gemini":
                # ...existing gemini setup code...
                genai = _llm_clients['gemini']
                genai.configure(api_key=args.gemini_key)

                # Note: More advanced safety settings and generation configs can be added here
                generation_config = genai.types.GenerationConfig(
                    # candidate_count=1, # Default is 1
                    # temperature=0.7 # Example temperature setting
                )
                # Safety settings can be adjusted if needed, e.g., to be less restrictive
                safety_settings = {
                    # Example: Block fewer things (use with caution)
                    # genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    # genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                }

                model = genai.GenerativeModel(
                    args.llm_model,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                    # system_instruction=system_message # Use this if model/API supports it directly
                    )
                # Send the effective prompt (potentially with system message prepended)
                response = model.generate_content(effective_prompt)

                # Handle potential blocks or errors
                if not response.candidates:
                     # Check if it was blocked due to safety/other reasons
                     try:
                         block_reason = response.prompt_feedback.block_reason
                         block_message = response.prompt_feedback.block_reason_message
                         print(f"Warning: Gemini call blocked. Reason: {block_reason} - {block_message}")
                     except Exception:
                         print("Warning: Gemini call returned no candidates and no block reason found.")
                     # Raise an exception to trigger retry or failure
                     raise ValueError("Gemini response blocked or empty")

                content = response.text # Access text directly from the main response object
                # log_verbose(f"LLM Response: {content[:200]}...") # Replaced by full echo below
                # return content.strip() # Return happens after echo

            # --- Echo Response ---
            if content is not None:
                print("\n--- LLM Response ---")
                print(content.strip())
                print("--- End LLM Response ---")
                return content.strip()
            else:
                # Handle cases where content might be None unexpectedly (e.g., Gemini block)
                # The exception handling below will catch actual errors/blocks
                raise ValueError("LLM response content was None unexpectedly.")
            # --- End Echo Response ---


        except Exception as e:
            # Specific handling for Gemini API errors if needed
            if args.llm_provider == "gemini" and "API key not valid" in str(e):
                 print(f"Fatal Gemini Error: Invalid API Key. Please check --gemini-key or GEMINI_API_KEY. ({e})")
                 exit(1) # Exit immediately for invalid key

            print(f"Error calling LLM on attempt {attempt + 1}/{args.max_retries}: {e}") # Use resolved value from args
            if attempt < args.max_retries - 1: # Use resolved value from args
                print(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                print("Max retries reached. LLM call failed.")
                return None
    return None

# --- NEW: Function to query webui.db ---
def get_collection_names_for_tag(db_path: str, tag_name: str) -> List[str]:
    """
    Finds Chroma collection names associated with a knowledge base name
    by querying the 'knowledge' table, extracting 'file_ids' from its 'data' JSON field,
    and prepending 'file-' to each ID.
    """
    collection_names = []
    # Rename tag_name internally for clarity, as it refers to the knowledge base name
    knowledge_base_name = tag_name
    log_verbose(f"Querying webui.db at '{db_path}' for knowledge base '{knowledge_base_name}'...")
    conn = None # Initialize conn to None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # --- Step 1: Find the knowledge base entry by name and get its 'data' field ---
        knowledge_entry_data = None
        try:
            # Only select the 'data' column now
            kb_query = "SELECT data FROM knowledge WHERE name = ?;"
            cursor.execute(kb_query, (knowledge_base_name,))
            kb_result = cursor.fetchone()
            if kb_result:
                knowledge_entry_data_json = kb_result[0]
                log_verbose(f"Found knowledge base entry for '{knowledge_base_name}'. Extracting file_ids...")

                # Attempt to parse JSON data and extract file_ids
                if knowledge_entry_data_json:
                    try:
                        knowledge_entry_data = json.loads(knowledge_entry_data_json)
                        # --- ACTUAL EXTRACTION LOGIC ---
                        if isinstance(knowledge_entry_data, dict) and 'file_ids' in knowledge_entry_data:
                            if isinstance(knowledge_entry_data['file_ids'], list):
                                # *** MODIFICATION HERE: Prepend "file-" to each ID ***
                                collection_names = [f"file-{str(fid)}" for fid in knowledge_entry_data['file_ids']] # Ensure they are strings and prepend
                                log_verbose(f"Successfully extracted and formatted {len(collection_names)} collection names (e.g., 'file-UUID') from knowledge.data.")
                            else:
                                print(f"Warning: 'file_ids' key found in knowledge.data for '{knowledge_base_name}', but its value is not a list.")
                        else:
                            print(f"Warning: Could not find 'file_ids' key within the JSON data for knowledge base '{knowledge_base_name}'.")
                        # --- END EXTRACTION LOGIC ---
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse 'data' JSON for knowledge base '{knowledge_base_name}': {e}")
                        print(f"Raw data: {knowledge_entry_data_json}")
                else:
                     print(f"Warning: 'data' column is empty for knowledge base '{knowledge_base_name}'.")

            else:
                print(f"Error: Knowledge base named '{knowledge_base_name}' not found in the 'knowledge' table.")
                # Exit here as we can't proceed without the knowledge base entry
                return [] # Return empty list

        except sqlite3.Error as e:
            print(f"Error finding/querying knowledge base entry for '{knowledge_base_name}': {e}")
            return [] # Return empty list on error

        # --- Step 2: Query is no longer needed here ---

    except sqlite3.Error as e:
        # General error during connection or initial setup
        print(f"An SQLite error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while accessing webui.db: {e}")
    finally:
        if conn:
            conn.close() # Ensure connection is closed

    # Return the extracted collection names (if found in JSON) or an empty list
    if not collection_names:
         # This message might appear if extraction failed or data was empty
         print(f"\nWarning: Failed to extract any collection names for knowledge base '{knowledge_base_name}'.")

    # Return the list (might be empty if extraction failed)
    return collection_names

# --- NEW: Function to Re-initialize ChromaDB Client ---
def reinitialize_chromadb_client(current_client, current_st_ef, args, core_doc):
    """
    Attempts to release resources by deleting and recreating the ChromaDB client
    and embedding function. Returns new instances or exits on failure.
    """
    log_verbose("Re-initializing ChromaDB client and embedding function...")
    start_reinit_time = time.time()
    new_client = None
    new_st_ef = None
    new_query_embedding = None

    # Explicitly delete old client and ef to hint GC
    try:
        del current_client
        del current_st_ef
    except NameError:
        pass # Should not happen in normal flow
    gc.collect() # Suggest garbage collection
    time.sleep(0.5) # Small pause

    try:
        new_client = chromadb.PersistentClient(path=args.db_path)
        new_st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=args.embed_model)
        # Re-embed core_doc as the embedding function instance changed
        new_query_embedding = new_st_ef([core_doc])[0]
        end_reinit_time = time.time()
        print(f"ChromaDB client re-initialized successfully in {end_reinit_time - start_reinit_time:.2f} seconds.")
        return new_client, new_st_ef, new_query_embedding
    except Exception as e:
        print(f"\nFATAL: Error re-initializing ChromaDB client mid-run: {e}")
        exit(1) # Exit if re-initialization fails

# --- Main Script Logic ---
def main():
    # --- Start Timer ---
    start_time = datetime.datetime.now()
    print(f"Script started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    # --- End Start Timer ---

    # --- Add Debugging ---
    print("--- Debugging Configuration ---")
    print(f"Value for --db-path: '{args.db_path}' (Type: {type(args.db_path)})")
    print(f"Value for --webui-db-path: '{args.webui_db_path}' (Type: {type(args.webui_db_path)})") # Added
    print(f"Value for --tag-name: '{args.tag_name}' (Type: {type(args.tag_name)})") # Added
    print(f"Value for --embed-model: '{args.embed_model}' (Type: {type(args.embed_model)})")
    print(f"Value for --llm-provider: {args.llm_provider} (Type: {type(args.llm_provider)})")
    print(f"Value for --llm-model: {args.llm_model} (Type: {type(args.llm_model)})")
    # --- Updated Debug Output ---
    print(f"Value for --num-results-per-coll: {args.num_results_per_coll} (Type: {type(args.num_results_per_coll)})") # Updated default
    print(f"Value for --max-context-docs: {args.max_context_docs} (Type: {type(args.max_context_docs)})")
    # print(f"Value for --max-distance-threshold: {args.max_distance_threshold} (Type: {type(args.max_distance_threshold)})") # Removed
    # --- End Updated Debug Output ---
    print(f"Value for --max-retries: {args.max_retries} (Type: {type(args.max_retries)})")
    print("-----------------------------")
    # --- End Debugging ---

    # 1. Connect to Vector DB Client (Initial Connection)
    log_verbose(f"Connecting to ChromaDB client at path: {args.db_path}")
    try:
        client = chromadb.PersistentClient(path=args.db_path)
        st_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=args.embed_model)
        log_verbose(f"Using embedding model: {args.embed_model}")
    except Exception as e:
        print(f"Error connecting to ChromaDB client: {e}")
        exit(1)

    # 1.b Get List of Potential Collection Names (from webui.db)
    potential_collection_names = get_collection_names_for_tag(args.webui_db_path, args.tag_name)
    if not potential_collection_names:
         # Error message already printed in get_collection_names_for_tag if needed
         print(f"Exiting as no potential collection names were found for knowledge base '{args.tag_name}'.")
         exit(1)
    log_verbose(f"Found {len(potential_collection_names)} potential collections listed in webui.db for '{args.tag_name}'.")

    # 1.c Filter against existing ChromaDB collections
    try:
        log_verbose("Fetching list of actual collections from ChromaDB...")
        existing_chroma_collections = client.list_collections()
        log_verbose(f"Found {len(existing_chroma_collections)} collections in ChromaDB.")

        # Find the intersection
        valid_collection_names = list(set(potential_collection_names) & set(existing_chroma_collections))

        if not valid_collection_names:
            print(f"Error: None of the {len(potential_collection_names)} collections listed for knowledge base '{args.tag_name}' in webui.db exist in ChromaDB at path '{args.db_path}'.")
            print("Please check for inconsistencies or indexing issues.")
            exit(1)

        log_verbose(f"Found {len(valid_collection_names)} valid collections common to webui.db and ChromaDB.")
        # Log removed collections if verbose
        if args.verbose:
             removed_count = len(potential_collection_names) - len(valid_collection_names)
             if removed_count > 0:
                  log_verbose(f"Ignoring {removed_count} collections listed in webui.db but not found in ChromaDB.")

    except Exception as e:
        print(f"Error listing or filtering ChromaDB collections: {e}")
        exit(1)

    # --- Outer Pipeline Retry Loop ---
    pipeline_attempt = 0
    success = False # Flag to track if we succeeded
    while pipeline_attempt < MAX_PIPELINE_ATTEMPTS:
        pipeline_attempt += 1
        print(f"\n--- Starting Pipeline Attempt {pipeline_attempt}/{MAX_PIPELINE_ATTEMPTS} ---")

        # --- Collection Sampling (Inside Loop) ---
        collections_to_search = valid_collection_names # Start with the full list for sampling
        total_valid_collections = len(valid_collection_names)
        sampled_collections_for_this_attempt = []

        if MAX_COLLECTIONS > 0 and MAX_COLLECTIONS < total_valid_collections:
            log_verbose(f"MAX_COLLECTIONS set to {MAX_COLLECTIONS}. Randomly sampling from {total_valid_collections} valid collections for attempt {pipeline_attempt}.")
            sampled_collections_for_this_attempt = random.sample(valid_collection_names, MAX_COLLECTIONS)
            log_verbose(f"Will search {len(sampled_collections_for_this_attempt)} randomly selected collections this attempt.")
        elif MAX_COLLECTIONS > 0:
            log_verbose(f"MAX_COLLECTIONS ({MAX_COLLECTIONS}) is >= total valid collections ({total_valid_collections}). Searching all valid collections.")
            sampled_collections_for_this_attempt = valid_collection_names # Use all
        else:
            log_verbose("MAX_COLLECTIONS is 0. Searching all valid collections.")
            sampled_collections_for_this_attempt = valid_collection_names # Use all
        # --- End Collection Sampling ---

        # 2. Get a Random Starting Point (from the *sampled* collections for this attempt)
        core_id = None
        core_doc = None
        core_meta = None
        selected_collection_name = None
        MAX_START_DOC_ATTEMPTS = 10 # Try up to 10 times within the sampled set
        start_doc_attempts = 0
        # Use a copy of the *sampled* list for attempts
        attempt_collection_list = sampled_collections_for_this_attempt.copy()

        log_verbose(f"Attempting to select a random starting document from {len(attempt_collection_list)} sampled collections (max {MAX_START_DOC_ATTEMPTS} attempts)...")
        while start_doc_attempts < MAX_START_DOC_ATTEMPTS:
            start_doc_attempts += 1
            log_verbose(f"Start doc attempt {start_doc_attempts}/{MAX_START_DOC_ATTEMPTS}...")
            try:
                if not attempt_collection_list:
                     log_verbose("Warning: Ran out of sampled collections to attempt for starting document in this pipeline attempt.")
                     break # Exit inner loop, will fail core_doc check below

                random_collection_name = random.choice(attempt_collection_list)
                log_verbose(f"Trying collection: {random_collection_name}")

                try: collection = client.get_collection(name=random_collection_name, embedding_function=st_ef)
                except Exception as e:
                     print(f"Warning: Sampled collection '{random_collection_name}' unexpectedly failed to load: {e}. Removing from attempts.")
                     attempt_collection_list.remove(random_collection_name)
                     continue

                results = collection.get(limit=10, include=['documents', 'metadatas'])
                if not results or not results.get('ids'):
                    log_verbose(f"Warning: Collection {random_collection_name} exists but appears empty. Removing from attempts.")
                    attempt_collection_list.remove(random_collection_name)
                    continue

                random_index = random.choice(range(len(results['ids'])))
                core_id = results['ids'][random_index]
                core_doc = results['documents'][random_index]
                core_meta = results['metadatas'][random_index] if results['metadatas'] else {}
                selected_collection_name = random_collection_name
                log_verbose(f"Selected core document ID: {core_id} from collection {selected_collection_name} (Source File: {core_meta.get('filename', 'N/A')})")
                log_verbose(f"Core document content (start): {core_doc[:200]}...")
                break # Exit inner loop successfully

            except Exception as e:
                print(f"Unexpected error during starting document selection (Attempt {start_doc_attempts}): {e}")
                if random_collection_name in attempt_collection_list: attempt_collection_list.remove(random_collection_name)
                time.sleep(1)
                continue

        if not core_doc:
            print(f"Error: Failed to find a valid starting document in sampled collections for pipeline attempt {pipeline_attempt}. Trying next attempt.")
            continue # Go to the next iteration of the outer pipeline loop

        # 3. Semantic Context Retrieval (from the *sampled* collections for this attempt)
        top_results_heap = []
        processed_doc_hashes = {hash(core_doc)}

        try:
            total_collections_to_search = len(sampled_collections_for_this_attempt) # Use the length of the sampled list
            log_verbose(f"Performing semantic search across {total_collections_to_search} sampled collections...")
            reinit_msg = f"Client re-initialization is ENABLED (every {CLIENT_REINIT_INTERVAL} collections)." if ENABLE_CLIENT_REINIT else "Client re-initialization is DISABLED."
            log_verbose(reinit_msg)
            log_verbose(f"Retrieving up to {args.num_results_per_coll} chunks per collection.")
            log_verbose(f"Maintaining top {args.max_context_docs} unique docs overall.")
            query_embedding = st_ef([core_doc])[0]

            # --- Use the sampled list for the loop ---
            for idx, collection_name in enumerate(sampled_collections_for_this_attempt):
                # --- Optional Periodic Re-initialization ---
                if ENABLE_CLIENT_REINIT and idx > 0 and idx % CLIENT_REINIT_INTERVAL == 0:
                    client, st_ef, query_embedding = reinitialize_chromadb_client(
                        client, st_ef, args, core_doc
                    )
                # --- End Optional Periodic Re-initialization ---

                # --- Calculate Elapsed Time ---
                current_time = datetime.datetime.now()
                elapsed_time = current_time - start_time
                total_seconds = int(elapsed_time.total_seconds())
                hours, remainder = divmod(total_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                elapsed_str = f"{hours:02}:{minutes:02}:{seconds:02}"
                # --- End Calculate Elapsed Time ---

                percentage = (idx + 1) / total_collections_to_search * 100
                # --- Update Progress Indicator ---
                progress_line = f"\rSearching collection {idx + 1}/{total_collections_to_search} ({percentage:.1f}%) | Found: {len(top_results_heap)}/{args.max_context_docs} | Elapsed: {elapsed_str}..."
                print(progress_line, end='')
                # --- End Update Progress Indicator ---

                try:
                    collection = client.get_collection(name=collection_name, embedding_function=st_ef)
                    search_result = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=args.num_results_per_coll, # Use updated default (3)
                        include=['documents', 'metadatas', 'distances']
                    )

                    # --- Process results using heapq ---
                    # ... (heap processing and notification logic remains the same) ...
                    if search_result and search_result.get('ids') and search_result['ids'][0]:
                        for i in range(len(search_result['ids'][0])):
                            distance = search_result['distances'][0][i]
                            document = search_result['documents'][0][i]
                            metadata = search_result['metadatas'][0][i] if search_result['metadatas'] else {}
                            doc_hash = hash(document)

                            if doc_hash in processed_doc_hashes: continue

                            result_item = { "id": search_result['ids'][0][i], "document": document, "distance": distance, "metadata": metadata, "collection": collection_name }

                            new_top_chunk_found = False
                            if len(top_results_heap) < args.max_context_docs:
                                heapq.heappush(top_results_heap, (-distance, result_item))
                                processed_doc_hashes.add(doc_hash)
                                if len(top_results_heap) == args.max_context_docs: new_top_chunk_found = True
                            elif distance < -top_results_heap[0][0]:
                                removed_item = heapq.heapreplace(top_results_heap, (-distance, result_item))
                                processed_doc_hashes.remove(hash(removed_item[1]['document']))
                                processed_doc_hashes.add(doc_hash)
                                new_top_chunk_found = True

                            if new_top_chunk_found:
                                print(f"\r{' ' * 120}\r", end='')
                                source_file = metadata.get('filename', 'N/A')
                                print(f"Found new top chunk (Dist: {distance:.4f}, File: {source_file})")
                                progress_line = f"\rSearching collection {idx + 1}/{total_collections_to_search} ({percentage:.1f}%) | Found: {len(top_results_heap)}/{args.max_context_docs} | Elapsed: {elapsed_str}..."
                                print(progress_line, end='')
                    # --- End Process results ---

                except Exception as e:
                    print(f"\r{' ' * 120}\r", end='')
                    log_verbose(f"Could not query collection {collection_name}: {e}")
                    continue

            print() # Newline after loop

            # --- Extract and Dynamically Determine Threshold ---
            # ... (knee point detection logic remains the same) ...
            final_top_results_sorted = sorted([item[1] for item in top_results_heap], key=lambda x: x['distance'])
            dynamic_distance_threshold = 1.5
            if len(final_top_results_sorted) >= 3:
                log_verbose(f"Attempting to find knee point in {len(final_top_results_sorted)} sorted distances...")
                distances = [r['distance'] for r in final_top_results_sorted]
                indices = list(range(len(distances)))
                try:
                    kneedle = KneeLocator(indices, distances, curve='convex', direction='increasing', S=1.0)
                    if kneedle.knee is not None:
                        knee_index = kneedle.knee
                        dynamic_distance_threshold = distances[knee_index]
                        log_verbose(f"Found knee point at index {knee_index} with distance: {dynamic_distance_threshold:.4f}")
                    else: log_verbose("Could not find a distinct knee point. Using default threshold.")
                except Exception as e: log_verbose(f"Error during knee point detection: {e}. Using default threshold.")
            else: log_verbose(f"Too few results ({len(final_top_results_sorted)}) to determine knee point. Using default threshold.")

            # --- Filter Final Results using Dynamic Threshold ---
            # ... (filtering logic remains the same) ...
            final_context_docs = [core_doc]
            docs_added_count = 1
            log_verbose(f"Filtering {len(final_top_results_sorted)} potential context chunks by dynamic distance <= {dynamic_distance_threshold:.4f}...")
            for result in final_top_results_sorted:
                 if docs_added_count >= args.max_context_docs:
                     log_verbose(f"Reached max context docs ({args.max_context_docs}).")
                     break
                 if result['document'] == core_doc: continue
                 if result['distance'] <= dynamic_distance_threshold:
                     final_context_docs.append(result['document'])
                     docs_added_count += 1
                 else:
                     log_verbose(f"Stopping filter: Next chunk distance ({result['distance']:.4f}) exceeds dynamic threshold ({dynamic_distance_threshold:.4f}).")
                     break
            combined_context = "\n\n---\n\n".join(final_context_docs)
            log_verbose(f"Combined context includes {len(final_context_docs)} unique document chunks after filtering.")
            log_verbose(f"Combined context (start): {combined_context[:300]}...")
            # --- End Extract and Filter Final Results ---

        except Exception as e:
            print(f"Error during multi-collection semantic search for attempt {pipeline_attempt}: {e}")
            continue # Go to the next iteration of the outer pipeline loop

        # --- LLM Pipeline ---
        try:
            # 4. Context Validation (LLM Call #1)
            print("\nStep 1: Validating Context...")
            validation_prompt = CONTEXT_VALIDATION_PROMPT_TEMPLATE.format(context=combined_context)
            validation_result = call_llm(validation_prompt, CONTEXT_VALIDATION_SYSTEM_PROMPT)
            if not validation_result: raise ValueError("Failed to get context validation result from LLM.")
            print(f"Context Validation Result: {validation_result}")
            if not validation_result.lower().startswith("sufficient"):
                print("Context deemed insufficient. Trying next pipeline attempt.")
                continue # Go to the next iteration of the outer pipeline loop

            # 5. Interpret and Explain (LLM Call #2)
            print("\nStep 2: Interpreting Script and Generating Explanation...")
            interpretation_prompt = INTERPRETATION_PROMPT_TEMPLATE.format(context=combined_context)
            explanation = call_llm(interpretation_prompt, INTERPRETATION_SYSTEM_PROMPT)
            if not explanation: raise ValueError("Failed to get explanation from LLM.")
            log_verbose(f"Generated Explanation:\n{explanation}")

            # 6. Explanation Review (LLM Call #3)
            print("\nStep 3: Reviewing Explanation...")
            review_prompt = EXPLANATION_REVIEW_PROMPT_TEMPLATE.format(script_context=combined_context, explanation=explanation)
            explanation_review = call_llm(review_prompt, EXPLANATION_REVIEW_SYSTEM_PROMPT)
            if not explanation_review: raise ValueError("Failed to get explanation review from LLM.")
            print(f"Explanation Review Result: {explanation_review}")
            if not explanation_review.lower().startswith("clear and accurate"):
                print("Explanation failed review. Trying next pipeline attempt.")
                continue # Go to the next iteration of the outer pipeline loop

            # 7. Generate Q&A (LLM Call #4)
            print("\nStep 4: Generating Question & Answer Pair...")
            qa_gen_prompt = QA_GENERATION_PROMPT_TEMPLATE.format(explanation=explanation)
            qa_json_str = call_llm(qa_gen_prompt, QA_GENERATION_SYSTEM_PROMPT)
            if not qa_json_str: raise ValueError("Failed to get Q&A JSON from LLM.")
            try:
                if qa_json_str.startswith("```json"): qa_json_str = qa_json_str[7:]
                if qa_json_str.endswith("```"): qa_json_str = qa_json_str[:-3]
                qa_pair = json.loads(qa_json_str.strip())
                if "question" not in qa_pair or "answer" not in qa_pair: raise ValueError("Generated JSON missing 'question' or 'answer' key.")
                log_verbose(f"Generated Q&A: {qa_pair}")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error processing generated Q&A JSON: {e}")
                print(f"LLM Output was: {qa_json_str}")
                raise ValueError("Failed to process Q&A JSON.") # Raise to trigger outer loop retry

            # 8. Q&A Quality/Relevance Review (LLM Call #5)
            print("\nStep 5: Reviewing Q&A Pair...")
            qa_review_prompt = QA_REVIEW_PROMPT_TEMPLATE.format(explanation=explanation, question=qa_pair["question"], answer=qa_pair["answer"])
            qa_review = call_llm(qa_review_prompt, QA_REVIEW_SYSTEM_PROMPT)
            if not qa_review: raise ValueError("Failed to get Q&A review from LLM.")
            print(f"Q&A Review Result: {qa_review}")

            # 9. Output and Save (Success!)
            if qa_review.lower().startswith("high quality"):
                print("\n--- Generated High-Quality Q&A Pair ---")
                print(json.dumps(qa_pair, indent=2))
                print("--- End of Pair ---")

                # --- Append to JSON file ---
                try:
                    output_filename = "ck3_qa_pairs.jsonl"
                    with open(output_filename, 'a', encoding='utf-8') as f:
                        # Write the JSON object as a single line
                        json.dump(qa_pair, f, ensure_ascii=False)
                        f.write('\n') # Add a newline to separate JSON objects (JSON Lines format)
                    print(f"Successfully appended Q&A pair to {output_filename}")
                except Exception as e:
                    print(f"Error writing Q&A pair to file '{output_filename}': {e}")
                # --- End Append to JSON file ---

                success = True # Set success flag
                break # Exit the while pipeline_attempt loop
            else:
                print("\nQ&A pair failed quality review. Trying next pipeline attempt.")
                continue # Go to the next iteration of the outer pipeline loop

        except Exception as llm_pipeline_error:
             # Catch errors during the LLM steps (including failed calls returning None)
             print(f"Error during LLM pipeline (Attempt {pipeline_attempt}): {llm_pipeline_error}")
             # Continue to the next iteration of the outer loop
             continue

    # --- After the Loop ---
    if not success: # Check the success flag
         print(f"\nFailed to generate a high-quality Q&A pair after {MAX_PIPELINE_ATTEMPTS} attempts.")

    # --- End Timer ---
    end_time = datetime.datetime.now()
    total_duration = end_time - start_time
    print(f"\nScript finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total execution time: {total_duration}")
    # --- End End Timer ---

if __name__ == "__main__":
    main()