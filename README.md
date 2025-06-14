# 📝 Multilingual Caption Aligner & Semantic Validator

This project translates image captions into a target language (e.g., Urdu, Spanish, Chinese) and validates that the translated captions retain their original meaning using sentence embeddings.

## 🔧 Features
- Translates image-caption pairs using Hugging Face translation pipelines
- Computes cosine similarity using LaBSE multilingual embeddings
- Filters out semantically inconsistent translations
- Exports aligned data to CSV

## 📂 Output Format
```csv
image_path,en_caption,lang_caption,similarity
💡 Technologies Used
Hugging Face Transformers

SentenceTransformers (LaBSE)

Python

📊 Example
Caption: "A dog running through the field"
Translation: "ایک کتا میدان سے چلّاتا ہے"
Similarity: 0.864
