import random
import pickle
import os
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer

# ------------------------------
# RL Chatbot Agent (Q-Learning)
# ------------------------------
class RLChatbot:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.actions = actions  # possible responses
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration rate
        self.q_table = defaultdict(lambda: np.zeros(len(actions)))
        self.vectorizer = CountVectorizer()

    def featurize(self, text):
        """Convert input text to a hashed state ID (string key)."""
        return str(hash(text.lower()) % (10**8))

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, len(self.actions) - 1)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state):
        """Q-learning update."""
        old_q = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        self.q_table[state][action] += self.alpha * (reward + self.gamma * next_max - old_q)

    def save(self, path="rl_chatbot.pkl"):
        with open(path, "wb") as f:
            pickle.dump((dict(self.q_table), self.actions), f)

    def load(self, path="rl_chatbot.pkl"):
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), data[0])
                self.actions = data[1]

# ------------------------------
# Simulated training environment
# ------------------------------
def simulated_reward(user_input, bot_response):
    """Fake reward function for simulation:  
       Higher reward if bot_response 'matches' intent."""
    if "hello" in user_input.lower() and "hello" in bot_response.lower():
        return 5
    if "bye" in user_input.lower() and "bye" in bot_response.lower():
        return 5
    if "help" in user_input.lower() and "help" in bot_response.lower():
        return 5
    return -1  # default negative reward

# ------------------------------
# Main program
# ------------------------------
if __name__ == "__main__":
    # actions = [
    #     "Hello! How can I help you?",
    #     "Goodbye! Have a nice day.",
    #     "I can help with your problems. What do you need?",
    #     "I'm not sure I understand.",
    #     "Please tell me more."
    # ]

    actions = [
    # General greetings & casual
    "Hello! How can I help you today?",
    "Hi there! What’s on your mind?",
    "Goodbye! Have a great day.",
    "See you later! Keep coding.",
    "I’m here to help with your questions.",

    # AI/ML related
    "Are you working on machine learning today?",
    "Which model architecture are you using?",
    "Do you want to discuss prompt engineering or fine-tuning?",
    "I can explain how transformers work in detail.",
    "Would you like me to write example PyTorch code for you?",
    "I can help debug your reinforcement learning agent.",
    "What dataset are you using for your project?",
    "Let’s talk about optimizing training performance.",
    "Are you running your model on CPU or GPU?",
    "I can guide you on hyperparameter tuning.",

    # Developer workflow
    "Would you like me to generate example code?",
    "I can help write a FastAPI endpoint for your AI model.",
    "Do you need help with Hugging Face Transformers?",
    "We can integrate this with a Flask web app.",
    "Do you want me to explain FAISS indexing?",
    "I can walk you through a RAG (Retrieval-Augmented Generation) pipeline.",
    "Let’s debug your Python code step-by-step.",
    "Would you like me to explain gradient descent?",

    # More conversational fallback
    "I’m not sure I understand, could you rephrase?",
    "Can you provide more details?",
    "Let’s break down the problem together.",
    "Interesting question! Let’s explore it.",
    "I can provide documentation links if you need.",
    "That’s a complex topic, but I can simplify it for you."
]



    agent = RLChatbot(actions)

    # ------------------------------
    # Simulated training phase
    # ------------------------------
    # training_data = [
    #     "hello", "hi there", "bye", "goodbye", "i need help", "can you help me",
    #     "what's up", "please help", "bye bye", "see you"
    # ]
    training_data = [
    # Greetings / casual
    "hello", "hi there", "hey", "good morning", "good evening",
    "what's up", "how are you", "yo", "long time no see", "how's it going",

    # General help
    "i need help", "can you help me", "please help", "i have a question",
    "i'm stuck", "can you guide me", "how do i fix this", "explain this to me",
    "can you give me an example", "show me sample code",

    # AI / ML specific
    "how to train a model", "what is reinforcement learning",
    "how does fine tuning work", "what is transfer learning",
    "explain gradient descent", "how to improve accuracy",
    "what is overfitting", "what is prompt engineering",
    "how to load a huggingface model", "how to use pytorch",
    "how to deploy a model", "difference between supervised and unsupervised learning",

    # Coding / debugging
    "why is my code not working", "how to debug python code",
    "what does this error mean", "how to fix module not found error",
    "how to install requirements", "what is virtual environment",
    "how to use git", "how to clone a repository",
    "what is docker", "how to run flask app",

    # Farewells
    "bye", "goodbye", "see you", "bye bye", "take care", "catch you later"
]


    for episode in range(200):
        user_msg = random.choice(training_data)
        state = agent.featurize(user_msg)
        action = agent.choose_action(state)
        bot_reply = actions[action]
        reward = simulated_reward(user_msg, bot_reply)
        next_state = agent.featurize("end")  # stateless
        agent.update(state, action, reward, next_state)

    print("✅ Training completed (simulated)")

    # Save trained model
    agent.save()

    # ------------------------------
    # Interactive chat
    # ------------------------------
    print("\n🤖 RL Chatbot is ready! Type 'quit' to exit.")
    agent.load()

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        state = agent.featurize(user_input)
        action = agent.choose_action(state)
        bot_reply = actions[action]
        print(f"Bot: {bot_reply}")

        # Get human feedback (reward)
        try:
            reward = int(input("Rate this reply (-5 to 5): "))
        except ValueError:
            reward = 0  # default if invalid
        next_state = agent.featurize("end")
        agent.update(state, action, reward, next_state)
        agent.save()

    print("💾 Chatbot model updated and saved.")
