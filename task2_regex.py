# Загружаем класс TextVectorizer
from text_vectorizer import TextVectorizer

docs = [
    "Я люблю программирование и изучать новое",
    "Программирование это интересно и полезно",
    "Изучать Python особенно приятно"
]

vectorizer = TextVectorizer(docs)

word = "программирование"
doc_index = 0

print(f"TF('{word}', doc {doc_index}):", vectorizer.get_tf(word, doc_index))
print(f"IDF('{word}'):", vectorizer.get_idf(word))
print(f"TF-IDF('{word}', doc {doc_index}):", vectorizer.get_tf_idf(word, doc_index))
print(f"TF-IDF('и', doc {doc_index}, ignore_stopwords=False):",
      vectorizer.get_tf_idf("и", doc_index, ignore_stopwords=False))
print(f"TF-IDF('и', doc {doc_index}, ignore_stopwords=True):",
      vectorizer.get_tf_idf("и", doc_index, ignore_stopwords=True))

# ------

# Загружаем функции из ulysses_analyzer
from ulysses_analyzer import load_ulysses_chapter, clean_and_count_words, find_word_context

text = load_ulysses_chapter()

if text:
    print("\nПервые 500 символов текста:")
    print(text[:500])

    counts = clean_and_count_words(text)
    print("\nТоп-20 слов по частоте:")
    for word, count in counts.most_common(20):
        print(f"{word}: {count}")

    find_word_context(text, "love", left_len=5, right_len=5, cut_length=True)
else:
    print("Не удалось загрузить текст.")