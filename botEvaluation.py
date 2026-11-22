from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch

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
    best_idx = int(torch.argmax(cos_scores))
    max_sim = float(cos_scores[best_idx])
    best_reference = reference_answers[best_idx]

    # Fast-fail if completely off-topic
    if max_sim < 0.50:
        return {
            "score": 0.0,
            "status": "Incorrecto (Fuera de tema)",
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
        status = "Incorrecto (Contradicción detectada)"
    elif max_sim >= 0.80 and entailment_prob > 0.50:
        score = 1.0
        status = "Correcto (Automático)"
    elif max_sim >= 0.65:
        score = 0.5
        status = "Parcial / Revisión manual"
    else:
        score = 0.0
        status = "Insuficiente / Revisión manual"

    return {
        "score": score,
        "status": status,
        "similarity": round(max_sim, 3),
        "entailment_prob": round(entailment_prob, 3),
        "contradiction_prob": round(contradiction_prob, 3)
    }

if __name__ == "__main__":
    # Example usage
    #Pregunta: "¿Qué es un valor promedio?"
    references = [
        "La fotosíntesis es el proceso mediante el cual las plantas convierten la luz solar en energía química.",
        "Es la transformación de energía luminosa en energía química que realizan los organismos autótrofos."
    ]

    test_cases = [
        "Proceso donde las plantas usan la luz del sol para crear energía.",          # Good paraphrase
        "La fotosíntesis no usa la luz solar para producir energía.",                  # Negation trap
        "Es algo importante que hacen las hojas verdes en el bosque.",                 # Too vague
        "La capital de Francia es París."                                              # Off-topic
    ]

    for ans in test_cases:
        result = evaluate_spanish_definition(ans, references)
        print(f"Respuesta: {ans}\nResultado: {result}\n{'-'*50}")