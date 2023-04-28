import nltk
import wikipedia
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data
nltk.download('punkt')

# Get the text content of a Wikipedia page
try:
    page = wikipedia.page("Miscegenation")
    text = page.content
except wikipedia.exceptions.PageError:
    print("Sorry, the requested Wikipedia page does not exist.")
    exit()
except wikipedia.exceptions.DisambiguationError:
    print("Sorry, the requested term is ambiguous. Please be more specific.")
    exit()
except wikipedia.exceptions.WikipediaException:
    print("Sorry, something went wrong while retrieving the Wikipedia page.")
    exit()

# Preprocess the text by converting to lowercase and concatenating tokens
tokens = nltk.word_tokenize(text.lower())
preprocessed_text = ' '.join(tokens)

# Chunk the preprocessed text into smaller chunks of 100 words each
chunk_size = 100
chunk_texts = [preprocessed_text[i:i + chunk_size] for i in range(0, len(preprocessed_text), chunk_size)]

# Load the DistilBERT model and tokenizer
model_name = 'distilbert-base-uncased'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertModel.from_pretrained(model_name)

# Set the device to use for the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Tokenize and encode the text chunks
chunk_encodings = tokenizer(chunk_texts, padding=True, truncation=True, return_tensors='pt')
chunk_encodings = chunk_encodings.to(device)

# Generate embeddings for the text chunks
with torch.no_grad():
    chunk_outputs = model(**chunk_encodings)
    chunk_embeddings = chunk_outputs.last_hidden_state[:, 0, :]

# Prompt the user for input
user_input = input("Enter your text: ")

# Preprocess the user input by converting to lowercase and tokenizing
user_input_tokens = nltk.word_tokenize(user_input.lower())
preprocessed_user_input = ' '.join(user_input_tokens)

# Tokenize and encode the user input
user_input_encoding = tokenizer(preprocessed_user_input, padding=True, truncation=True, return_tensors='pt')
user_input_encoding = user_input_encoding.to(device)

# Generate an embedding for the user input
with torch.no_grad():
    user_input_output = model(**user_input_encoding)
    user_input_embedding = user_input_output.last_hidden_state[:, 0, :]

# Compute cosine similarities between the user input and each text chunk
similarities = cosine_similarity(user_input_embedding.cpu().numpy(), chunk_embeddings.cpu().numpy())[0]

# Find the most similar chunk
matching_chunk_indices = [i for i in range(len(similarities)) if similarities[i] > 0.1]

if len(matching_chunk_indices) == 0:
    print("Sorry, no matching chunks found.")
else:
    most_similar_chunk_index = matching_chunk_indices[similarities[matching_chunk_indices].argmax()]
    most_similar_chunk = chunk_texts[most_similar_chunk_index]
    accuracy_percentage = round(similarities[most_similar_chunk_index] * 100, 2)

    print(f"The most similar chunk to the user input is:\n{most_similar_chunk}")
    print(f"The accuracy percentage is: {accuracy_percentage}%")
    print(f"Out of {len(chunk_texts)} chunks, {len(matching_chunk_indices)} matched")
