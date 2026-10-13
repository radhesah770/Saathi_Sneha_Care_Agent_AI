import os
from typing import TypedDict, Optional

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from faq_agent import check_escalation
from faq_agent import faq_data

#Load API key from the project root .env file
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

#Build a more controlled FAQ matcher
faq_questions = []
faq_answers = []

for faq in faq_data:
    for question in faq["questions"]:
        faq_questions.append(question)
        faq_answers.append(faq["answer"])

#Add common paraphrases missing from the original FAQ data
extra_questions = [
    (
         "How can I book a caregiver?",
        "You can contact Saathi Sneha Care and provide information "
        "about your parent's care needs. The care team can help you "
        "choose a suitable care plan and arrange the required support."
    ),
    (
          "How do I hire a caregiver?",
        "You can contact Saathi Sneha Care and provide information "
        "about your parent's care needs. The care team can help you "
        "choose a suitable care plan and arrange the required support."
    ),
]

for question, answer in extra_questions:
    faq_questions.append(question)
    faq_answers.append(answer)

faq_vectorizer = TfidfVectorizer(stop_words="english")
faq_matrix = faq_vectorizer.fit_transform(faq_questions)


def find_relevant_faq(question: str) -> Optional[str]:
    """Return an FAQ answer only when the question matches appropriately."""

    unsupported_topics = {
        "mars","moon","planet","space","juipter","saturn" 
    }

    question_words = set(question.lower().replace("?", "").split())

    if question_words.intersection(unsupported_topics):
        return None

    query_vector = faq_vectorizer.transform([question])
    scores = cosine_similarity(query_vector, faq_matrix)[0]

    best_index = scores.argmax()
    best_score = scores[best_index]

    matched_question = faq_questions[best_index]

    # Ignore words that are common and don't help identify the topic.
    common_words = {
        "a", "an", "the", "is", "are", "do", "does", "did",
        "i", "you", "we", "they", "what", "how", "can", "could",
        "would", "should", "to", "for", "of", "on", "in", "my",
        "me", "your", "our", "and", "or", "please", "provide"
    }

    query_words = set(question.lower().replace("?", "").split())
    matched_words = set(matched_question.lower().split())

    important_query_words = {
        word.strip(".,!?")
        for word in query_words
        if word.strip(".,!?") not in common_words
    }

    important_matched_words = {
        word.strip(".,!?")
        for word in matched_words
        if word.strip(".,!?") not in common_words
    }

    # Require at least one meaningful word from the FAQ question.
    if not important_query_words.intersection(important_matched_words):
        return None

    # Reject weak matches.
    if best_score < 0.30:
        return None

    print(f"[DEBUG] Best FAQ similarity: {best_score:.3f}")
    print(f"[DEBUG] Matched FAQ question: {matched_question}")

    return faq_answers[best_index]




class SupportState(TypedDict):
    question: str
    response: str
    escalation_category: Optional[str]
    needs_escalation: bool

def check_request(state: SupportState) -> dict:
    """Check safety and escalation rules before answering."""
    category, message = check_escalation(state["question"])

    if category:
        return{
            "response": message,
            "escalation_category": category,
            "needs_escalation": True,
        }

    return {
        "response": "",
        "escalation_category": None,
        "needs_escalation": False,
    }

def answer_question(state: SupportState) -> dict:
    """Use a relevant FAQ first, then Gemini for appropriate fallback."""

    # 1. Try a high-confidence FAQ match.
    answer = find_relevant_faq(state["question"])

    if answer:
        return {"response": answer}

    # 2. Use Gemini only when the FAQ does not match confidently.
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            temperature=0,
        )

        faq_context = "\n".join(
            f"- {faq['answer']}" for faq in faq_data
        )

        result = llm.invoke(
            "You are the Saathi Sneha Care customer-support assistant.\n"
            "Use only the following approved FAQ information:\n"
            f"{faq_context}\n\n"
            "Rules:\n"
            "- Do not invent services, policies, prices, or payment methods.\n"
            "- If the FAQ does not answer the question, say you cannot "
            "confirm the information and recommend contacting support.\n"
            "- Never provide diagnoses, medication recommendations, "
            "or medical advice.\n"
            "- Be polite and concise.\n\n"
            f"Customer question: {state['question']}"
        )

        content = result.content

        if isinstance(content,list):
            text_parts = [
                item["text"]
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            content = "\n".join(text_parts)
        return {"response": content}

    except Exception:
        return {
            "response": (
                "I'm sorry, I couldn't find a reliable answer. "
                "Please contact Saathi Sneha Care support for assistance."
            )
        }

def route_after_check(state: SupportState) -> str:
    if state["needs_escalation"]:
        return "end"
    return "answer"

def build_graph():
    graph = StateGraph(SupportState)

    graph.add_node("check_request", check_request)
    graph.add_node("answer", answer_question)

    graph.add_edge(START, "check_request")

    graph.add_conditional_edges(
        "check_request",
        route_after_check,
        {
            "end": END,
            "answer": "answer",
        },
    )

    graph.add_edge("answer", END)

    return graph.compile()

support_graph = build_graph()

def get_support_response(question: str) -> dict:
    """Run the support workflow for one customer question."""
    result = support_graph.invoke(
        {
            "question": question,
            "response": "",
            "escalation_category": None,
            "needs_escalation": False,
        }
    )

    return result

if __name__ == "__main__":
    print("Saathi Sneha Care - LangGraph Support Agent")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        result = get_support_response(question)

        print(f"\nAssistant: {result['response']}")

        if result["needs_escalation"]:
            print(
                f"[Escalation required: ]"
                f"{result['escalation_category']}"
            )

        print()