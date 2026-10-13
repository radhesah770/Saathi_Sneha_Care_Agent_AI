from faq_data import faq_data

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Prepare FAQ questions

all_questions = []
question_to_faq = []

for faq in faq_data:

    for question in faq["questions"]:

        all_questions.append(question)
        question_to_faq.append(faq)


# Create TF-IDF vectors

vectorizer = TfidfVectorizer(stop_words="english")

faq_vectors = vectorizer.fit_transform(all_questions)

# Escalation Check

def check_escalation(user_question):

    question = user_question.lower().strip()


    
    # Emergency keywords
    

    emergency_words = [
        "emergency",
        "unconscious",
        "not breathing",
        "can't breathe",
        "cannot breathe",
        "heart attack",
        "stroke",
        "severe bleeding",
        "serious injury",
        "accident",
        "dying"
    ]

    if any(word in question for word in emergency_words):

        return (
            "EMERGENCY",
            "This sounds like an emergency. "
            "Please contact emergency services immediately. "
            "I will also escalate this conversation to a human "
            "support representative."
        )


    
    # Medical keywords
    

    medical_words = [
        "medicine",
        "medication",
        "tablet",
        "drug",
        "doctor",
        "diagnosis",
        "symptom",
        "symptoms",
        "pain",
        "fever",
        "bleeding",
        "injury",
        "medical",
        "health problem",
        "prescription"
    ]

    if any(word in question for word in medical_words):

        return (
            "MEDICAL",
            "This question involves a medical issue, "
            "so I should not provide medical advice. "
            "I will escalate this conversation to a human "
            "support representative."
        )

    #Billing keywords
    billing_keywords = [
        "charged twice",
        "double charged",
        "duplicate payment",
        "incorrect charge",
        "wrong charge",
        "billing dispute",
        "payment deducted twice"
    ]

    if any(keyword in question.lower() for keyword in billing_keywords):
        return (
            "BILLING",
            "Your billing concern should be reviewed by a human support representative. I will escalate this conversation to the support team"
        )
    
    # Angry customer / serious complaint
    

    angry_words = [
        "angry",
        "furious",
        "terrible service",
        "worst service",
        "very bad service",
        "hate your service",
        "disappointed",
        "unhappy",
        "complaint",
        "complain",
        "lawsuit",
        "legal action",
        "refund now",
        "want to speak to a manager",
        "speak to a human",
        "speak to a person",
        "human agent"
    ]

    if any(word in question for word in angry_words):

        return (
            "COMPLAINT",
            "I'm sorry that you're experiencing this issue. "
            "This situation should be handled by a human "
            "support representative. I will escalate your "
            "conversation to the support team."
        )


    
    # No escalation
    

    return None, None


# Find answer

def find_answer(user_question):

    question = user_question.lower().strip()

    #step 1 : check for escalation
    escalation_type, escalation_message = check_escalation(
        user_question
    )

    if escalation_type:
        return escalation_message
    
    # Payment intent

    payment_words = [
        "payment",
        "khalti",
        "stripe"
    ]

    if any(word in question for word in payment_words):

        for faq in faq_data:

            if "How can I pay?" in faq["questions"]:

                return faq["answer"]


    # "pay" with payment-related phrases
    if "pay" in question:

        payment_phrases = [
            "how can i pay",
            "how do i pay",
            "make a payment",
            "payment method",
            "where can i pay"
        ]

        if any(phrase in question for phrase in payment_phrases):

            for faq in faq_data:

                if "How can I pay?" in faq["questions"]:

                    return faq["answer"]


    
    # TF-IDF similarity search
    

    user_vector = vectorizer.transform([user_question])

    similarities = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_index = similarities.argmax()

    best_score = similarities[0][best_index]


    
    # Minimum similarity
    

    if best_score >= 0.20:

        matched_faq = question_to_faq[best_index]

        return matched_faq["answer"]


    
    # No answer
    

    return (
        "I'm sorry, I couldn't find an answer to your question. "
        "Please contact Saathi Sneha Care support for further assistance."
    )


# Start Support Agent
def main():
    print("=====================================")
    print("   Saathi Sneha Care Support Agent")
    print("=====================================")
    print("Type 'exit' to end the conversation.")


    while True:

        question = input("\nCustomer: ")


        # Exit
        if question.lower().strip() == "exit":

            print(
                "Agent: Thank you for contacting "
                "Saathi Sneha Care. Goodbye!"
            )

            break


        # Find answer
        answer = find_answer(question)


        # Display answer
        print("Agent:", answer)
if __name__ == "__main__":
    main()