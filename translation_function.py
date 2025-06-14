from transformers import pipeline

def get_translation_pipeline(target_lang="ur"):
    model_map = {
        "ur": "Helsinki-NLP/opus-mt-en-ur",
        "es": "Helsinki-NLP/opus-mt-en-es",
        "zh": "Helsinki-NLP/opus-mt-en-zh"
    }
    model_name = model_map.get(target_lang)
    if not model_name:
        raise ValueError("Unsupported language code.")
    return pipeline("translation", model=model_name)

def translate_captions(captions, target_lang="ur"):
    translator = get_translation_pipeline(target_lang)
    return [translator(caption, max_length=512)[0]["translation_text"] for caption in captions]