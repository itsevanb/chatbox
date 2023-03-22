import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

#lodaing GPT-2 and tokenizer 
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

#create a function to generate respones
def generate_response(input_text, max_length=100, num_return_sequences=1): 
#max_length - 100 tokens for generated response, r_n_s - model should only generate one sequence 
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
#tokenizes input_text, output should be a pytorch tensor, a pyt-tensor is a data structure similar #to an array or vector -- a container for numerical data that you can perform mathematical #operations on, utilizes GPU instead of CPU 
    output_sequence = model.generate( 
#calls generate method on GPT-2 model and takes number of paramerters
        input_ids = input_ids,
        max_length = max_length,
        num_return_sequences = num_return_sequences,
        no_repeat_ngram_size = 2, 
#size of ngrams to avoid repeating in generated text/helps prevent repetitve phrases
#n-grams are contiguous(words or characters that appear one after the other) sequence of n items #from a sample of text or speech
        do_sample = True,
        temperature = 0.7,
#controls randomness of generated text, higher values the more random lower values more deterministic
        top_k = 50,
#reduces probablity of generating low probability tokens
        top_p = 0.95,
#uses nucleus sampling(more creative and natural sounding output)(picks next token based on #probability threshold rather than always picking most likely token)threshold or nucleus is the #probability value that determines how many possible next words to consider, rather than looking at #most probably tokens(top_k sampling\\top_p is subset of top_k)
    )
    decoded_output = [tokenizer.decode(sequence) for sequence in output_sequence]
#decodes generated token sequences back into readible text iterating through out_putsequences
    return decoded_output

#command-line interface for chatbot

if __name__ == '__main__':
    print(f'\033[92m''Welcome to ChatBox :)\nTo exit please enter "quit" ''\033[0m')

    while True: 
#creates an infinite loop unless user input 'quit'
        user_input = input('\033[91m'"User:") 
        if user_input.lower() == 'quit':
            break
        chatbot_response = generate_response(user_input)[0]
#calls generate response function and passes 'user_input' as an argument since generate response #returns a list of responses the '[0]' only returns the first reposnse from the list
        print(f"'\033[94m'Evbot: {chatbot_response}'\033[0m'")