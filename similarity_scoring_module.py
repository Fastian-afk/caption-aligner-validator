from sentence_transformers import SentenceTransformer, util

def compute_similarity(en_captions, translated_captions):
    model = SentenceTransformer("sentence-transformers/LaBSE")
    en_embeds = model.encode(en_captions, convert_to_tensor=True)
    tr_embeds = model.encode(translated_captions, convert_to_tensor=True)
    similarities = util.cos_sim(en_embeds, tr_embeds)
    return [float(similarities[i][i]) for i in range(len(en_captions))]