import pandas as pd
import numpy as np
import re
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
import warnings
import time
from multiprocessing import Value

warnings.filterwarnings("ignore", category=FutureWarning)


shared_tfidf_count = None    # Race Condition #1
shared_word_count = None     # Race Condition #2



def init_shared_vars(stc, swc):
    """
    Registers shared counter variables inside each worker process.
    Without this, macOS keeps them as 0.
    """
    global shared_tfidf_count, shared_word_count
    shared_tfidf_count = stc
    shared_word_count = swc



def clean_text(text):
    """Clean individual text entry."""
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text))
    return text.lower()


def clean_text_chunk(chunk_data):
    """Clean chunk of text."""
    chunk_df = chunk_data.copy()
    chunk_df["clean_text"] = chunk_df["text"].apply(clean_text)
    return chunk_df[["clean_text"]]


def extract_genres(genres_str):
    """Extract genres from string representation."""
    try:
        genres = ast.literal_eval(genres_str)
        if isinstance(genres, (set, list)):
            return list(genres)
        else:
            return []
    except:
        return []


def vectorize_chunk(texts):
    """
    TF-IDF chunk (Race #1 happens here)
    """
    texts = [str(t) if pd.notna(t) else "" for t in texts]
    
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20)
    tfidf_matrix = vectorizer.fit_transform(texts)


    for _ in range(tfidf_matrix.shape[0]):
        temp = shared_tfidf_count.value
        time.sleep(0.0001) 
        shared_tfidf_count.value = temp + 1

    return tfidf_matrix, vectorizer.get_feature_names_out()


def count_words_chunk(texts):
    """
    Count word frequencies (Race #2 happens here)
    """
    texts = [str(t) if pd.notna(t) else "" for t in texts]
    all_text = " ".join(texts)

    for _ in range(len(texts)):
        temp = shared_word_count.value
        time.sleep(0.0001)  
        shared_word_count.value = temp + 1

    return Counter(all_text.split())


def count_genres_chunk(genre_lists):
    """Count genres in a chunk."""
    all_genres = [g for sublist in genre_lists for g in sublist]
    return Counter(all_genres)



def main():
    
    NUM_WORKERS = 8
    CHUNK_MULTIPLIER = 4  

    # Load
    print(" Loading dataset...")
    df = pd.read_csv("/Users/fellwakh/Downloads/newpar/books_and_genres.csv")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    print(f"Dataset loaded successfully: {len(df)} rows\n")

    # Initialize real shared counters
    stc = Value('i', 0)   # TF-IDF shared
    swc = Value('i', 0)   # Word count shared

    start_time = time.time()

    
    print(" Stage 1: Parallel Text Cleaning")
    chunk_size = len(df) // NUM_WORKERS
    chunks = [
        df.iloc[i:i+chunk_size] if i+chunk_size < len(df) else df.iloc[i:]
        for i in range(0, len(df), chunk_size)
    ][:NUM_WORKERS]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        cleaned_chunks = list(executor.map(clean_text_chunk, chunks))

    df["clean_text"] = pd.concat(cleaned_chunks, ignore_index=True)["clean_text"]
    print("Text cleaning completed.\n")

   
    print("Stage 2: Genre Extraction")
    df["genre_list"] = df["genres"].apply(extract_genres)
    print("Genre extraction completed.\n")


    print("Stage 3: Parallel TF-IDF Vectorization (with race conditions)")

    small_chunk_size = len(df) // (NUM_WORKERS * CHUNK_MULTIPLIER)
    text_chunks = [
        df["clean_text"].iloc[i:i+small_chunk_size].tolist() 
        if i+small_chunk_size < len(df) 
        else df["clean_text"].iloc[i:].tolist()
        for i in range(0, len(df), small_chunk_size)
    ]

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=init_shared_vars,
        initargs=(stc, swc)
    ) as executor:
        tfidf_results = list(executor.map(vectorize_chunk, text_chunks))

    tfidf_matrices = [result[0] for result in tfidf_results]
    final_tfidf = vstack(tfidf_matrices)

    all_tfidf_terms = set()
    for _, terms in tfidf_results:
        all_tfidf_terms.update(terms)

    print("TF-IDF vectorization completed.\n")


    print("Stage 4: Parallel Word Counting (with race conditions)")

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=init_shared_vars,
        initargs=(stc, swc)
    ) as executor:
        word_counters = list(executor.map(count_words_chunk, text_chunks))

    final_word_counter = Counter()
    for counter in word_counters:
        final_word_counter.update(counter)

    print("Word counting completed.\n")

    print("Stage 5: Parallel Genre Counting")

    genre_chunks = [
        df["genre_list"].iloc[i:i+small_chunk_size].tolist()
        if i+small_chunk_size < len(df) 
        else df["genre_list"].iloc[i:].tolist()
        for i in range(0, len(df), small_chunk_size)
    ]

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        genre_counters = list(executor.map(count_genres_chunk, genre_chunks))

    final_genre_counter = Counter()
    for counter in genre_counters:
        final_genre_counter.update(counter)

    print("Genre counting completed.\n")


    print("\n" + "=" * 60)
    print("RACE CONDITION EVIDENCE")
    print("=" * 60)
    print(f"Expected rows                    = {len(df)}")
    print(f"TF-IDF Processed (shared)        = {stc.value}")
    print(f"Word Count Processed (shared)    = {swc.value}")
    print(f"TF-IDF Loss                      = {len(df) - stc.value}")
    print(f"Word Count Loss                  = {len(df) - swc.value}")

    if stc.value != len(df) or swc.value != len(df):
        print("\n🔴 RACE CONDITION DETECTED!")
        print("   Multiple workers overwrote each other's increments.")
        print("   This is why you NEED locks/atomic operations in parallel code!")
    else:
        print("\n🟢 No race detected in this run (try running again!)")

  
    end_time = time.time()
    elapsed = end_time - start_time

    print("\n" + "=" * 60)
    print(" RESULTS")
    print("=" * 60)

    print("\n TOP 15 WORDS:")
    for word, count in final_word_counter.most_common(15):
        print(f"{word:<20} {count}")

    print("\n TOP 10 GENRES:")
    for genre, count in final_genre_counter.most_common(10):
        print(f"{genre:<20} {count}")

    print("\nTF-IDF TERMS (Across all chunks):")
    for i, term in enumerate(sorted(all_tfidf_terms), 1):
        print(f"{i}. {term}")

    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print("\n" + "=" * 60)
    print(" PERFORMANCE")
    print("=" * 60)
    print(f"Total rows processed: {len(df)}")
    print(f"Workers used: {NUM_WORKERS}")
    print(f"Chunks created: {len(text_chunks)}")
    print(f"Execution time: {minutes} min {seconds} sec")
    print(f"Avg time per row: {elapsed / len(df):.4f} sec")



if __name__ == '__main__':
    main()