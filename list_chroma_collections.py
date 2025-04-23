import os
import argparse
try:
    import chromadb
except ImportError:
    print("Error: chromadb library not found. Please install with 'pip install chromadb'")
    exit(1)
try:
    from dotenv import load_dotenv
except ImportError:
    pass # Optional

# --- Helper function to read and clean env vars ---
def get_clean_env_var(key, default=None):
    value = os.getenv(key, default)
    if value is None:
        return None
    cleaned_value = value.split('#')[0].strip().strip('\'"')
    return cleaned_value

# Load .env to get the path easily
load_dotenv()
DEFAULT_VECTOR_DB_PATH = get_clean_env_var("VECTOR_DB_PATH", "./backend/data/vector_db") # Default if not in .env

parser = argparse.ArgumentParser(description="List collections in a ChromaDB instance.")
parser.add_argument("--db-path", default=DEFAULT_VECTOR_DB_PATH, help="Path to the ChromaDB database directory.")
args = parser.parse_args()

if not args.db_path:
    print("Error: DB Path is required. Set VECTOR_DB_PATH in .env or use --db-path argument.")
    exit(1)

print(f"Attempting to connect to ChromaDB at: {args.db_path}")

try:
    client = chromadb.PersistentClient(path=args.db_path)
    # list_collections() now returns a list of names (strings) in ChromaDB >= 0.6.0
    collection_names = client.list_collections()

    if not collection_names:
        print("No collections found in this ChromaDB instance.")
    else:
        print("\nFound Collections:")
        # Iterate through the list of names directly
        for name in collection_names:
            print(f"- {name}")

except Exception as e:
    print(f"\nError connecting to or listing collections in ChromaDB: {e}")
    print(f"Ensure the path '{args.db_path}' is correct and points to a valid ChromaDB directory.")
