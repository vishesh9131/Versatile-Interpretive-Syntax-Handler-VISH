import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import os
import json
import time

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define TransformerEncoder class
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(rate)
        self.dropout2 = nn.Dropout(rate)

    def forward(self, x):
        attn_output, _ = self.att(x, x, x)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

# Define TextDataset class
class TextDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# Define function to build model
def build_model(vocab_size, embedding_dim, max_length):
    class TransformerModel(nn.Module):
        def __init__(self, vocab_size, embedding_dim, max_length):
            super(TransformerModel, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, max_length)
            self.transformer_encoder1 = TransformerEncoder(embed_dim=embedding_dim, num_heads=8, ff_dim=512)
            self.transformer_encoder2 = TransformerEncoder(embed_dim=embedding_dim, num_heads=8, ff_dim=512)
            self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(embedding_dim, vocab_size)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, x):
            x = self.embedding(x)
            x = x.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim) -> (seq_len, batch_size, embed_dim)
            x = self.transformer_encoder1(x)
            x = self.transformer_encoder2(x)
            x = x.permute(1, 2, 0)  # (seq_len, batch_size, embed_dim) -> (batch_size, embed_dim, seq_len)
            x = self.global_avg_pool(x).squeeze(-1)
            x = self.fc(x)
            return self.softmax(x)

    return TransformerModel(vocab_size, embedding_dim, max_length)

# Define function to load data
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    text = text.split('\n')
    return text

# Define function to preprocess data
def preprocess_data(text, max_length):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences, tokenizer

# Define Tokenizer class
class Tokenizer:
    def __init__(self):
        self.word_index = {}
        self.index_word = {}
        self.num_words = 0

    def fit_on_texts(self, texts):
        word_freq = {}
        for text in texts:
            words = text.split()
            for word in words:
                if word not in word_freq:
                    word_freq[word] = 1
                else:
                    word_freq[word] += 1
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        self.word_index = {word: idx + 1 for idx, (word, _) in enumerate(sorted_words)}
        self.index_word = {idx: word for word, idx in self.word_index.items()}
        self.num_words = len(self.word_index) + 1

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            sequences.append([self.word_index.get(word, 0) for word in text.split()])
        return sequences

# Define function to pad sequences
def pad_sequences(sequences, maxlen, padding='post'):
    padded_sequences = np.zeros((len(sequences), maxlen), dtype=int)
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            padded_sequences[i] = seq[:maxlen]
        else:
            if padding == 'post':
                padded_sequences[i, :len(seq)] = seq
            elif padding == 'pre':
                padded_sequences[i, -len(seq):] = seq
    return padded_sequences

# Define function to generate text with animation and metrics
def generate_text(model, tokenizer, seed_text, max_length, num_words, device, temperature=1.0, top_p=0.9):
    model.eval()
    seed_sequence = tokenizer.texts_to_sequences([seed_text])[0]
    generated_text = seed_text

    # Create placeholders for the warning box and generated text
    metrics_placeholder = st.empty()
    text_placeholder = st.empty()

    start_time = time.time()
    first_token_time = None

    for i in range(num_words):
        padded_sequence = pad_sequences([seed_sequence], maxlen=max_length, padding='post')
        padded_sequence = torch.tensor(padded_sequence, dtype=torch.long).to(device)
        with torch.no_grad():
            predicted_probs = model(padded_sequence).cpu().numpy()[0]

        # Apply temperature
        predicted_probs = np.log(predicted_probs + 1e-9) / temperature
        predicted_probs = np.exp(predicted_probs) / np.sum(np.exp(predicted_probs))

        # Top-p (nucleus) sampling
        sorted_indices = np.argsort(predicted_probs)[::-1]
        cumulative_probs = np.cumsum(predicted_probs[sorted_indices])
        top_p_indices = sorted_indices[cumulative_probs <= top_p]
        if len(top_p_indices) == 0:
            top_p_indices = sorted_indices[:1]
        top_p_probs = predicted_probs[top_p_indices]
        top_p_probs = top_p_probs / np.sum(top_p_probs)
        predicted_word_index = np.random.choice(top_p_indices, p=top_p_probs)

        predicted_word = tokenizer.index_word.get(predicted_word_index, '')

        if predicted_word == '':
            break

        if first_token_time is None:
            first_token_time = time.time() - start_time

        seed_sequence.append(predicted_word_index)
        seed_sequence = seed_sequence[1:]
        generated_text += ' ' + predicted_word

        # Update the text in the Streamlit app
        # Update metrics
        elapsed_time = time.time() - start_time
        tokens_per_sec = (i + 1) / elapsed_time

        # Update the warning box with metrics
        metrics_placeholder.success(
            f"""
            **SEC TO FIRST TOKEN:** {first_token_time:.2f} SEC  
            **SEC:** {elapsed_time:.2f} SEC  
            **TOKENS/SEC:** {tokens_per_sec:.2f}
            """
        )

        text_placeholder.markdown(f"<p style='font-size:12px;'>{generated_text}</p>", unsafe_allow_html=True)
        time.sleep(0.001) 

    return generated_text, first_token_time, elapsed_time, tokens_per_sec

# Define function to train model
def train_model(model, train_loader, criterion, optimizer, num_epochs, model_path):
    for epoch in range(num_epochs):
        model.train()
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        st.write(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Save the model
    torch.save(model.state_dict(), model_path)
    st.write("Model saved to disk.")

# Define function to evaluate model
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            test_loss += criterion(outputs, labels).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    # st.write(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Streamlit app
def main():
    st.title("VISH gpt")
    st.info("\"This is currently in the testing phase, but it’s already live and accessible on Streamlit Cloud.\" -vishesh")
    # Hardcoded values
    filepath = 'data_1.txt'
    max_length = 100
    embedding_dim = 96
    num_epochs = 20

    if not os.path.exists(filepath):
        st.error("File not found. Please enter a valid file path.")
        return

    text = load_data(filepath)
    random.shuffle(text)
    split_idx = int(0.8 * len(text))
    train_text, test_text = text[:split_idx], text[split_idx:]

    # Load the tokenizer if it exists
    if os.path.exists('tokenizer.json'):
        with open('tokenizer.json', 'r') as f:
            word_index = json.load(f)
        tokenizer = Tokenizer()
        tokenizer.word_index = word_index
        tokenizer.index_word = {v: k for k, v in word_index.items()}
        X_train_pad = pad_sequences(tokenizer.texts_to_sequences(train_text), maxlen=max_length, padding='post')
        X_test_pad = pad_sequences(tokenizer.texts_to_sequences(test_text), maxlen=max_length, padding='post')
    else:
        X_train_pad, tokenizer = preprocess_data(train_text, max_length)
        X_test_pad, _ = preprocess_data(test_text, max_length)
        # Save the tokenizer
        with open('tokenizer.json', 'w') as f:
            json.dump(tokenizer.word_index, f)

    vocab_size = len(tokenizer.word_index) + 1

    y_train = np.random.randint(vocab_size, size=len(X_train_pad))
    y_test = np.random.randint(vocab_size, size=len(X_test_pad))

    train_dataset = TextDataset(X_train_pad, y_train)
    test_dataset = TextDataset(X_test_pad, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = build_model(vocab_size, embedding_dim, max_length).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    model_path = 'VISH.pth'
    if os.path.exists(model_path):
        # Load the model if it exists and the vocab size matches
        try:
            model.load_state_dict(torch.load(model_path))
            model.eval()
            # st.write("Model loaded from disk.")
        except RuntimeError as e:
            st.write(f"Error loading model: {e}")
            st.write("Retraining the model due to vocabulary size mismatch.")
            train_model(model, train_loader, criterion, optimizer, num_epochs, model_path)
    else:
        # Train the model if it doesn't exist
        train_model(model, train_loader, criterion, optimizer, num_epochs, model_path)

    evaluate_model(model, test_loader, criterion, device)

    seed_text = st.text_input("Enter the seed text:", "Once upon a time")
    num_words = st.number_input("Enter the number of words to generate:", min_value=1, value=100, step=100)
    if st.button("Generate Text"):
        generated_text, first_token_time, elapsed_time, tokens_per_sec = generate_text(model, tokenizer, seed_text, max_length, num_words, device)


if __name__ == "__main__":
    main()