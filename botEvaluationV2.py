from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
import json

# 1. Load open-source models (downloaded automatically on first run)
# BGE-M3 excels at multilingual/Spanish semantic similarity
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# mDeBERTa NLI detects Entailment vs. Contradiction in Spanish
nli_model = CrossEncoder("MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

def evaluate_spanish_definition(student_answer: str, reference_answers: list[str]) -> dict:
    # Handle empty or overly short responses
    if not student_answer or len(student_answer.strip()) < 5:
        return {"score": 0.0, "status": "Reprobado (Vacío)", "similarity": 0.0, "contradiction_prob": 0.0}

    # Step 1: Semantic Similarity against all valid reference answers
    ref_embeddings = embedder.encode(reference_answers, convert_to_tensor=True)
    student_embedding = embedder.encode(student_answer, convert_to_tensor=True)
    
    cos_scores = util.cos_sim(student_embedding, ref_embeddings)[0]
    #print(f"Cosine Similarity Scores: {cos_scores.tolist()}")  # Debugging line to print similarity scores
    best_idx = int(torch.argmax(cos_scores))
    max_sim = float(cos_scores[best_idx])
    best_reference = reference_answers[best_idx]
    #print(f"Best Reference: {best_reference}, Max Similarity: {max_sim:.3f}")  # Debugging line to print best reference and similarity

    # Fast-fail if completely off-topic
    if max_sim < 0.50:
        return {            
            "status": "Incorrecto: la respuesta del estudiante está fuera de tema",
            "score": 0.0,
            "similarity": round(max_sim, 3),
            "contradiction_prob": 0.0
        }

    # Step 2: NLI Check (Premise = Reference, Hypothesis = Student Answer)
    # Model label mapping for MoritzLaurer/mDeBERTa-v3-base-mnli-xnli:
    # 0: entailment, 1: neutral, 2: contradiction
    nli_logits = nli_model.predict([(best_reference, student_answer)])[0]
    nli_probs = torch.softmax(torch.tensor(nli_logits), dim=0).tolist()
    
    entailment_prob = nli_probs[0]
    contradiction_prob = nli_probs[2]

    # Step 3: Decision Logic & Grading Buckets
    if contradiction_prob > 0.60:
        score = 0.0
        status = "Incorrecto: se detecta contradicción con la respuesta del estudiante"
    elif max_sim >= 0.80 and entailment_prob > 0.50:
        score = 1.0
        status = "Respuesta correcta de forma automatizada"
    elif max_sim >= 0.65:
        score = 0.5
        status = "Parcial: Realizar revisión manual"
    else:
        score = 0.0
        status = "Insuficiente información en la respuesta del estudiante para realizar una evaluación automática: Realizar revisión manual"

    return {
        "status": status,
        "score": score,        
        "similarity": round(max_sim, 3),
        "entailment_prob": round(entailment_prob, 3),
        "contradiction_prob": round(contradiction_prob, 3)
    }

if __name__ == "__main__":
    # Load data from JSON file
    with open("ejemplo_1.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Extract data from JSON
    pregunta = data.get("pregunta", "")
    references = [data.get("references", "")]
    
    print(f"Pregunta: {pregunta}\n")
    print(f"Referencias: {references[0]}\n")
    print("=" * 70)
    
    # Evaluate each student answer
    for key, value in data.items():
        if key.startswith("estudiante_"):
            student_name = key
            student_answer = value
            result = evaluate_spanish_definition(student_answer, references)
            print(f"\n{student_name}: {student_answer}")
            print(f"Resultado: {result}")
            print("-" * 70)