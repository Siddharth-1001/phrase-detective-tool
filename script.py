import nltk
import tensorflow_hub as hub
import wikipedia
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

page = wikipedia.page("Miscegenation")
text = page.content

tokens = nltk.word_tokenize(text.lower())
preprocessed_text = ' '.join(tokens)

chunk_size = 100
chunk_texts = [preprocessed_text[i:i + chunk_size] for i in range(0, len(preprocessed_text), chunk_size)]

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
chunk_embeddings = model(chunk_texts)

user_input = input("Enter your text: ")

user_input_tokens = nltk.word_tokenize(user_input.lower())
preprocessed_user_input = ' '.join(user_input_tokens)
user_input_embedding = model([preprocessed_user_input])[0]

similarities = cosine_similarity(user_input_embedding.numpy().reshape(1, -1), chunk_embeddings)[0]

matching_chunk_indices = [i for i in range(len(similarities)) if similarities[i] > 0.1]

if len(matching_chunk_indices) == 0:
    print("Sorry, no matching chunks found.")
else:
    most_similar_chunk_index = matching_chunk_indices[similarities[matching_chunk_indices].argmax()]
    most_similar_chunk = chunk_texts[most_similar_chunk_index]
    accuracy_percentage = round(similarities[most_similar_chunk_index] * 100, 2)

    print(f"The most similar chunk to the user input is:\n{most_similar_chunk}")
    print(f"The accuracy percentage is: {accuracy_percentage}%")
    print(f"Out of {len(chunk_texts)} chunks, {len(matching_chunk_indices)} matched the user input.")
