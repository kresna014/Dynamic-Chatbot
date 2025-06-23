from trasformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class DynamicChatbot:
    def __init__(self, personality="assistant"):
        self.model_name = "gpt2"
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.chat_history = []
        self.set_personality(personality)
        
    def set_personality(self, personality):
        personalities = {
            "assistant": "You are an assistant.",
            "friend": "Hey Buddy! How's it going?",
            "sarcastic": "Oh great, another human to talk to. How exciting."
        }
        self.personality = personalities.get(personality, "You are an assistant.")
        
    def generate_response(self, user_input):
        # gabungan riwayat percakapan + input baru
        prompt = f"{self.personality}\nChat History:\n"  "\n".join(self.chat_history[-5]) + f"\nUser: {user_input}\nBot:"
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs, max_lenght=150, num_return_sequences=1, pad_token_id=self.tokenizer.eos_token_id)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Hanya ambil bagian respons terakhir
        response = response.split("Bot:")[-1].strip()
        
        # Menyimpan riwayat percakapan
        self.chat_history.append(f"User: {user_input}")
        self.chat_history.append(f"Bot: {response}")
        
        return response
    
# Main program
if __name__ == "__main__":
    bot = DynamicChatbot(personality="friend") # ganti sarcastic atau assistant
    
    print("Bot: Hi! How can I help you today? (Type 'quit' to exit)")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response = bot.generate_response(user_input)
        print(f"Bot: {response}")