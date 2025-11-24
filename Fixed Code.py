
import pandas as pd
import re
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Value, Lock
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import time

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
#              CONFIG (EDIT THESE IF NEEDED)
# =========================================================
# Your dataset path:
CSV_PATH = r"C:books_and_genres.csv"

NUM_WORKERS = 4        # how many processes
MAX_ROWS = None       
TFIDF_MAX_FEATURES = 100
CHUNK_SIZE = 100
MODES = ["critical", "atomic", "reduction"]

# =========================================================
#              SHARED STATE FOR CRITICAL/ATOMIC
# =========================================================
shared_tfidf_count = None
shared_word_count = None
global_lock = None


def init_shared_vars(stc, swc, glock=None):
    global shared_tfidf_count, shared_word_count, global_lock
    shared_tfidf_count = stc
    shared_word_count = swc
    global_lock = glock


# =========================================================
#                    HELPER FUNCTIONS
# =========================================================
def clean_text(text):
    text = str(text)
    # truncate long texts to protect memory/time
    max_chars = 4000
    if len(text) > max_chars:
        text = text[:max_chars]
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    return text.lower()


def clean_text_chunk(chunk_df):
    chunk_df = chunk_df.copy()
    chunk_df["clean_text"] = chunk_df["text"].apply(clean_text)
    return chunk_df[["clean_text"]]


def extract_genres(genres_str):
    try:
        genres = ast.literal_eval(genres_str)
        if isinstance(genres, (set, list)):
            return list(genres)
        return []
    except Exception:
        return []


# =========================================================
#                    TF-IDF CHUNK
# =========================================================
def vectorize_chunk(texts, mode):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=TFIDF_MAX_FEATURES,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    local_rows = tfidf_matrix.shape[0]

    if mode == "critical":
        with global_lock:
            shared_tfidf_count.value += local_rows
    elif mode == "atomic":
        with shared_tfidf_count.get_lock():
            shared_tfidf_count.value += local_rows
    elif mode == "reduction":
        # reduction: no shared write
        pass

    # we only need local_rows, not the matrix
    return local_rows


def vectorize_chunk_wrapper(args):
    texts, mode = args
    return vectorize_chunk(texts, mode)


# =========================================================
#                    WORD COUNT CHUNK
# =========================================================
def count_words_chunk(texts, mode):
    all_text = " ".join(texts)
    local_rows = len(texts)

    if mode == "critical":
        with global_lock:
            shared_word_count.value += local_rows
    elif mode == "atomic":
        with shared_word_count.get_lock():
            shared_word_count.value += local_rows
    elif mode == "reduction":
        pass

    word_counts = Counter(all_text.split())
    return word_counts, local_rows


def count_words_chunk_wrapper(args):
    texts, mode = args
    return count_words_chunk(texts, mode)


# =========================================================
#                    GENRE COUNT CHUNK
# =========================================================
def count_genres_chunk(genre_lists):
    all_genres = [g for sublist in genre_lists for g in sublist]
    return Counter(all_genres)


# =========================================================
#                    MAIN PIPELINE
# =========================================================
def run_pipeline(mode, csv_path, num_workers=4, max_rows=None):
    """
    Run the whole pipeline once with a given synchronization mode.

    mode: 'critical', 'atomic', or 'reduction'
    Returns a dict with:
      - total time
      - TF-IDF time
      - word-count time
      - counters
    """
    assert mode in ("critical", "atomic", "reduction")

    print(f"\n======================== {mode.upper()} ========================")

    # ---------------- 1) Load dataset ----------------
    df = pd.read_csv(csv_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if "text" not in df.columns:
        raise ValueError("Expected a 'text' column in the dataset.")
    if "genres" not in df.columns:
        raise ValueError("Expected a 'genres' column in the dataset.")

    if max_rows is not None:
        df = df.head(max_rows)

    total_rows = len(df)
    print(f"Total rows used in this run: {total_rows}")

    total_start = time.time()   # total pipeline timer

    # ---------------- 2) Build chunks ----------------
    chunk_size = min(CHUNK_SIZE, total_rows)
    ranges = list(range(0, total_rows, chunk_size))
    chunks = [df.iloc[i:i + chunk_size] for i in ranges]

    # ---------------- 3) Cleaning ----------------
    print("Cleaning text in parallel...")
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        cleaned_chunks = list(exe.map(clean_text_chunk, chunks))
    df["clean_text"] = pd.concat(cleaned_chunks, ignore_index=True)["clean_text"]

    # ---------------- 4) Genres ----------------
    df["genre_list"] = df["genres"].apply(extract_genres)
    text_chunks = [df["clean_text"].iloc[i:i + chunk_size].tolist() for i in ranges]
    genre_chunks = [df["genre_list"].iloc[i:i + chunk_size].tolist() for i in ranges]

    # ---------------- 5) shared counters ----------------
    stc = Value("i", 0)
    swc = Value("i", 0)
    glock = Lock()

    # ---------------- 6) TF-IDF (timed) ----------------
    print("Running TF-IDF stage...")
    tfidf_start = time.time()

    tfidf_args = [(tc, mode) for tc in text_chunks]
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_shared_vars,
        initargs=(stc, swc, glock),
    ) as exe:
        tfidf_local_rows = list(exe.map(vectorize_chunk_wrapper, tfidf_args))

    tfidf_elapsed = time.time() - tfidf_start
    tfidf_min = tfidf_elapsed / 60.0

    # ---------------- 7) word count (timed) ----------------
    print("Running word-count stage...")
    word_start = time.time()

    word_args = [(tc, mode) for tc in text_chunks]
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_shared_vars,
        initargs=(stc, swc, glock),
    ) as exe:
        word_results = list(exe.map(count_words_chunk_wrapper, word_args))

    word_elapsed = time.time() - word_start
    word_min = word_elapsed / 60.0

    word_counters = [r[0] for r in word_results]
    word_local_rows = [r[1] for r in word_results]

    final_word_counter = Counter()
    for wc in word_counters:
        final_word_counter.update(wc)

    # ---------------- 8) genre count ----------------
    print("Running genre-count stage...")
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        genre_results = list(exe.map(count_genres_chunk, genre_chunks))

    final_genre_counter = Counter()
    for gc in genre_results:
        final_genre_counter.update(gc)

    # ---------------- 9) counters ----------------
    if mode == "reduction":
        tfidf_counter = sum(tfidf_local_rows)
        word_counter = sum(word_local_rows)
    else:
        tfidf_counter = stc.value
        word_counter = swc.value

    # ---------------- 10) total time ----------------
    total_elapsed = time.time() - total_start
    total_min = total_elapsed / 60.0

    tfidf_m = int(tfidf_elapsed // 60)
    tfidf_s = int(tfidf_elapsed % 60)
    word_m = int(word_elapsed // 60)
    word_s = int(word_elapsed % 60)
    total_m = int(total_elapsed // 60)
    total_s = int(total_elapsed % 60)

    print("\n--------- COUNTER CHECK ---------")
    print(f"Expected rows:   {total_rows}")
    print(f"TF-IDF count:    {tfidf_counter}")
    print(f"Word count:      {word_counter}")
    print(f"TF-IDF time:     {tfidf_m} min {tfidf_s} sec")
    print(f"Word-count time: {word_m} min {word_s} sec")
    print(f"Total time:      {total_m} min {total_s} sec")

    print("\nTop 10 words:")
    for w, c in final_word_counter.most_common(10):
        print(f"{w:<18} {c}")

    print("\nTop 10 genres:")
    for g, c in final_genre_counter.most_common(10):
        print(f"{g:<25} {c}")

    return {
        "mode": mode,
        "rows": total_rows,
        "time_min": total_min,
        "time_sec": total_elapsed,
        "tfidf_time_min": tfidf_min,
        "tfidf_time_sec": tfidf_elapsed,
        "word_time_min": word_min,
        "word_time_sec": word_elapsed,
        "tfidf_counter": tfidf_counter,
        "word_counter": word_counter,
    }
def main():
    results = []
    for mode in MODES:
        res = run_pipeline(
            mode=mode,
            csv_path=CSV_PATH,
            num_workers=NUM_WORKERS,
            max_rows=MAX_ROWS,
        )
        results.append(res)

    print("\n==================== SUMMARY ====================")
    print(
        f"{'Mode':<10} {'Rows':<8} "
        f"{'Total(s)':<10} {'TF-IDF(s)':<10} {'Word(s)':<10} "
        f"{'TF-IDF Cnt':<12} {'Word Cnt':<10}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['mode']:<10} "
            f"{r['rows']:<8} "
            f"{r['time_sec']:<10.2f} "
            f"{r['tfidf_time_sec']:<10.2f} "
            f"{r['word_time_sec']:<10.2f} "
            f"{r['tfidf_counter']:<12} "
            f"{r['word_counter']:<10}"
        )


if __name__ == "__main__":
    main()
