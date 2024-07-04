import openai
import csv
import logging
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from difflib import SequenceMatcher
from os.path import exists
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Hardcoded API key
openai.api_key = 'sk-llm-assignment2-oQO0w7Pc3wvHU1jwRUOTT3BlbkFJLZo9i9wxFvKbWswK21pf'

client = openai.OpenAI(api_key=openai.api_key)
# Intent classification setup
intents = ['order_status', 'human_representative', 'return_policy', 'general']
X = [
    "check my order", "order status", "where is my order", "track my package", "status of my order",
    "speak to a person", "human representative", "talk to someone", "need help from a person",
    "what's your return policy", "can I return", "refund policy", "return items", "return process",
    "hello", "hi", "what can you do", "help", "how are you", "good day"
]
y = [
    'order_status', 'order_status', 'order_status', 'order_status', 'order_status',
    'human_representative', 'human_representative', 'human_representative', 'human_representative',
    'return_policy', 'return_policy', 'return_policy', 'return_policy', 'return_policy',
    'general', 'general', 'general', 'general', 'general', 'general'
]

vectorizer = CountVectorizer()
classifier = MultinomialNB()
classifier.fit(vectorizer.fit_transform(X), y)

# Return policy information
return_policy_info = {
    "general": "You can return most items within 30 days of purchase for a full refund or exchange. Items must be in their original condition, with all tags and packaging intact.",
    "exceptions": "Certain items such as clearance merchandise, perishable goods, and personal care items are non-returnable.",
    "refund": "Refunds will be issued to the original form of payment. If you paid by credit card, the refund will be credited to your card. If you paid by cash or check, you will receive a cash refund."
}

# Load a pre-trained model for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_ai_response(prompt, conversation_history):
    """
    Generate an AI response based on the given prompt and conversation history.

    Args:
        prompt (str): The user's input prompt.
        conversation_history (List[Dict[str, str]]): The list of previous messages in the conversation.

    Returns:
        str: The generated AI response.

    Raises:
        Exception: If an error occurs while processing the request.

    Notes:
        - The AI assistant is a helpful assistant for an e-commerce platform.
        - It can assist with order status inquiries, connect customers with human representatives,
          and provide information about return policies.
        - The AI assistant should always be polite, concise, and helpful.
        - The AI assistant should only provide information relevant to e-commerce assistance.
        - If an error occurs while processing the request, a generic error message is returned.
    """
    try:
        system_message = {
            "role": "system",
            "content": (
               "You are a helpful AI assistant for an e-commerce platform. "
                "You can assist with order status inquiries, connect customers with human representatives, "
                "and provide information about return policies. Always be polite, concise, and helpful. "
                "When answering inquiries about order status, request the order number and provide relevant details. "
                "For requests to speak to a human representative, gather the user's name, email, and phone number. "
                "For return policy inquiries, provide information about return periods, conditions, and the refund process. "
                "Do not answer questions that are not related to e-commerce assistance."
                "If an error occurs while processing the request, a generic error message is returned."
                "If you answer a somewhat off topic thats somewhat related to e coomerce question, answer it shortly and get back to the conversation as an Ai assistant."
            )
        }
        messages = [system_message] + conversation_history + [{"role": "user", "content": prompt}]
        
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        full_reply_content= response.choices[0].message.content
         # Check the relevance of the response using intent classification
        if not is_relevant_response(full_reply_content):
            return "I'm sorry, I can only assist with inquiries related to our e-commerce platform. How can I help you with your order or our services?"

        return full_reply_content
    except Exception as e:
        logging.error(f"Error in get_ai_response: {e}")
        return "Sorry, I encountered an error while processing your request. Please try again later."

def classify_intent(query):
    """
    Classifies the intent of the query using a pre-trained classifier.

    Args:
        query (str): The input query to classify.

    Returns:
        str: The predicted intent of the query.
    """
    return classifier.predict(vectorizer.transform([query]))[0]

def read_order_status_from_csv(order_id):
    """
    Reads the order status from a CSV file based on the provided order_id.

    Args:
        order_id (str): The ID of the order to retrieve the status for.

    Returns:
        str: The status of the order corresponding to the given order_id. Returns None if the order_id is not found.

    Raises:
        Exception: If an error occurs while reading the order status from the CSV file.
    """
    try:
        with open('orders.csv', mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == order_id:
                    return row[1]
        return None
    except Exception as e:
        logging.error(f"Error reading order status from CSV: {e}")
        return None

def write_order_status_to_csv(order_id, status):
    """
    Writes the order status to a CSV file.

    Args:
        order_id (str): The ID of the order.
        status (str): The status of the order.

    Returns:
        None

    Raises:
        Exception: If an error occurs while writing the order status to the CSV file.
    """
    try:
        with open('orders.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([order_id, status])
    except Exception as e:
        logging.error(f"Error writing order status to CSV: {e}")


def setup_demo_database():
    demo_orders = [
        ("12345", "processed and expected to ship within the next 2 business days"),
        ("67890", "shipped and will be delivered by the end of the week"),
        ("54321", "delayed due to an issue with the payment method"),
        ("09876", "delivered"),
    ]
    try:
        with open('orders.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["order_id", "status"])  # Write header
            writer.writerows(demo_orders)
    except Exception as e:
        logging.error(f"Error setting up demo database: {e}")

def handle_order_status():
    """
    Retrieves the status of an order based on the provided order ID.

    Parameters:
        None

    Returns:
        str: A message indicating the status of the order. If the order is found,
             the message includes the order ID and its current status. If the order
             is not found, the message indicates that the order ID is invalid.
             If there is an error retrieving the order status, the message indicates
             the error and asks the user to try again later.
    """
    try:
        order_id = input("Please provide your order ID: ")
        status = read_order_status_from_csv(order_id)
        if status:
            return f"I've checked the status of order {order_id}. It is currently {status}. Is there anything else you'd like to know about this order?"
        else:
            return "Sorry, I couldn't find an order with that ID. Please check the order ID and try again."
    except Exception as e:
        logging.error(f"Error in handle_order_status: {e}")
        return "Sorry, I couldn't retrieve your order status at the moment. Please try again later."

def request_human_representative():
    """
    Requests a human representative and logs the contact information.

    This function prompts the user for their full name, email address, and phone number.
    It validates the email address and phone number and saves the contact information.
    It then returns a message indicating that a human representative will contact the user
    within the next 24 hours.

    Returns:
        str: A message indicating the successful request and the next steps.

    Raises:
        Exception: If an error occurs while processing the request.

    """
    try:
        name = input("Certainly, I'd be happy to connect you with a human representative. First, could you please provide your full name? ")
        email = input("Thank you. And your email address? ")
        while not validate_email(email):
            email = input("The email address you provided is not valid. Please enter a valid email address: ")
        
        phone = input("Lastly, what's the best phone number to reach you? ")
        while not validate_phone(phone):
            phone = input("The phone number you provided is not valid. Please enter a valid phone number: ")
        
        save_contact_info(name, email, phone)
        return f"Thank you, {name}. I've logged your request and a human representative will contact you within the next 24 hours at either {email} or {phone}. Is there anything else I can help you with while you wait?"
    except Exception as e:
        logging.error(f"Error in request_human_representative: {e}")
        return "Sorry, I couldn't connect you with a human representative at the moment. Please try again later."

def validate_email(email):
    """
    Validates an email address using a regular expression.

    Parameters:
        email (str): The email address to be validated.

    Returns:
        bool: True if the email address is valid, False otherwise.
    """
    regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(regex, email) is not None

def validate_phone(phone):
    """
    Validates a phone number using a regular expression.

    Parameters:
        phone (str): The phone number to be validated.

    Returns:
        bool: True if the phone number is valid, False otherwise.
    """
    regex = r'^\+?[1-9]\d{1,14}$'
    return re.match(regex, phone) is not None and len(phone) <= 15

def is_relevant_response(response):
    """
    Check if the response is relevant to e-commerce assistance using intent classification.

    Args:
        response (str): The response generated by the AI.

    Returns:
        bool: True if the response is relevant, False otherwise.
    """
    predicted_intent = classify_intent(response)
    return predicted_intent in intents

def save_contact_info(name, email, phone):
    """
    A function that saves contact information to a CSV file. It takes the name, email, and phone number as parameters.
    """
    try:
        file_exists = exists('contact_requests.csv')
        with open('contact_requests.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["name", "email", "phone"])  # Write header if file doesn't exist
            writer.writerow([name, email, phone])
    except Exception as e:
        logging.error(f"Error in save_contact_info: {e}")

def handle_return_policy(query):
    """
    Handles the return policy based on the given query.

    Args:
        query (str): The query to determine the return policy.

    Returns:
        str: The response message based on the query. If the query contains "cannot be returned", returns the exceptions message along with a prompt to specify a specific item. If the query contains "refund", returns the refund message along with a prompt to initiate a return or ask questions about the refund process. Otherwise, returns the general return policy message along with a prompt to ask specific questions about returns or exchanges. If an error occurs, returns an error message indicating that the return policy couldn't be retrieved and to try again later.
    """
    try:
        if "cannot be returned" in query.lower():
            return return_policy_info["exceptions"] + " Is there a specific item you're concerned about returning?"
        elif "refund" in query.lower():
            return return_policy_info["refund"] + " Would you like to initiate a return or do you have any other questions about the refund process?"
        else:
            return return_policy_info["general"] + " Do you have any specific questions about returns or exchanges?"
    except Exception as e:
        logging.error(f"Error in handle_return_policy: {e}")
        return "Sorry, I couldn't retrieve the return policy at the moment. Please try again later."

def handle_general_query(query, conversation_history):
    """
    A function that handles a general query by calling another function to get an AI response.

    Args:
        query: The query to be processed.
        conversation_history: The history of the conversation.

    Returns:
        The response generated by the AI assistant.
    """
    return get_ai_response(query, conversation_history)


def compute_similarity(response, expected_response):
    """
    Compute the similarity between the AI response and the expected response using semantic similarity.

    Args:
        response (str): The AI response.
        expected_response (str): The expected response.

    Returns:
        float: The similarity score between 0 and 1.
    """
    response_embedding = model.encode(response, convert_to_tensor=True)
    expected_response_embedding = model.encode(expected_response, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(response_embedding, expected_response_embedding)
    return cosine_sim.item()


def handle_user_session(user_id):
    """
    Handles a user session by interacting with the user and providing responses based on their input.

    Args:
        user_id (int): The ID of the user.

    Returns:
        None

    Raises:
        Exception: If an error occurs while handling the user session.

    Notes:
        - The function starts a session for the user and prints a greeting message.
        - It then enters a loop where it prompts the user for input and processes it.
        - The user input is classified into different intents using the `classify_intent` function.
        - Depending on the intent, the appropriate response is generated using the corresponding handler functions.
        - The conversation history is maintained in the `conversation_history` list.
        - If an error occurs during the session, an error message is printed and the loop continues.
    """
    conversation_history = []
    logging.info(f"Starting session for User {user_id}")
    print(f"AI to User {user_id}: Hello! How can I assist you today?")
    while True:
        try:
            user_input = input(f"User {user_id}: ")
            if user_input.lower() == 'exit':
                break
            
            conversation_history.append({"role": "user", "content": user_input})
            
            intent = classify_intent(user_input)
            
            if intent == 'order_status':
                response = handle_order_status()
            elif intent == 'human_representative':
                response = request_human_representative()
            elif intent == 'return_policy':
                response = handle_return_policy(user_input)
            else:
                response = handle_general_query(user_input, conversation_history)
            
            conversation_history.append({"role": "assistant", "content": response})
            
            print(f"\nAI to User {user_id}:", response)
        except Exception as e:
            logging.error(f"Error in handle_user_session: {e}")
            print("Sorry, an error occurred. Please try again.")
def predefined_dialogues():
    """
    Test predefined dialogues and evaluate performance metrics.
    """
    dialogues = [
        {
            "user": "check my order",
            "expected_response": "Of course! Could you please provide me with your order number so I can check the status for you?"
        },
        {
            "user": "speak to a person",
            "expected_response": "Certainly, I'd be happy to connect you with a human representative. May I have your name, email address, and phone number, please?"
        },
        {
            "user": "what's your return policy",
            "expected_response": "Our return policy allows for returns within 30 days of the delivery date. Items must be in their original condition with tags attached. Once we receive the returned item, a refund will be processed to the original payment method. If you have any specific questions about our return policy, feel free to ask."
        },
        {
            "user": "hello",
            "expected_response": "Hello! How can I assist you today?"
        }
    ]
    
    correct_responses = 0
    threshold = 0.65  # Define a similarity threshold
    for dialogue in dialogues:
        user_input = dialogue["user"]
        expected_response = dialogue["expected_response"]
        
        conversation_history = []
        response = get_ai_response(user_input, conversation_history)
        
        similarity_score = compute_similarity(response, expected_response)
        print(f"User: {user_input}\nAI: {response}\nExpected: {expected_response}\nSimilarity: {similarity_score:.2f}\n")
        
        if similarity_score >= threshold:
            correct_responses += 1
    
    accuracy = correct_responses / len(dialogues)
    print(f"Accuracy: {accuracy * 100:.2f}%")
def main():
    setup_demo_database()  # Set up the demo database
    predefined_dialogues() #test predefined dialogues

    num_users = 1  # Number of concurrent users
    for i in range(num_users):
        handle_user_session(i)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical(f"Critical error in main: {e}")