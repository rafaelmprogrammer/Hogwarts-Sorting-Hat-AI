import torch
from transformers import pipeline
import gradio as gr

# --- 1. SETUP: Initialize the AI Pipeline ---
# This part of the code initializes the language model.
# It will download the model the first time you run it.
model_name = "deepseek-ai/deepseek-coder-1.3b-instruct"
device = 0 if torch.cuda.is_available() else -1

print("Initializing the Sorting Hat pipeline...")
sorting_hat_pipeline = pipeline("text-generation", model=model_name, device=device)
print("Pipeline initialized successfully.")


# --- 2. LOGIC: The Core Sorting Function ---
def determine_house(
        answer1,
        answer2,
        answer3,
        answer4,
        answer5
):
    """
    Constructs the prompt for the model and determines the Hogwarts house.
    """
    initial_questions = [
        "What do you value most in a friend?",
        "If faced with a difficult choice, do you follow your head or your heart?",
        "What kind of challenges excite you the most?",
        "How do you react when you fail at something?",
        "What is your greatest fear?"
    ]
    user_answers = [answer1, answer2, answer3, answer4, answer5]

    # Construct the detailed prompt for the model
    prompt = "The following are questions and answers from a Hogwarts student being sorted:\n\n"
    for i in range(len(initial_questions)):
        prompt += f"Question: {initial_questions[i]}\nAnswer: {user_answers[i]}\n\n"

    # Final instruction for the model to generate the house name
    prompt += "Based on these answers, I, the Sorting Hat, must now place this student in the most suitable Hogwarts house. The house names are Gryffindor, Hufflepuff, Ravenclaw, or Slytherin. Be brief and state only the house name. The house is "

    # Call the text-generation pipeline with the constructed prompt
    generated_output = sorting_hat_pipeline(prompt, max_new_tokens=50, num_return_sequences=1)

    # Extract the generated house name from the model's output
    generated_text = generated_output[0]['generated_text']

    # New, more robust extraction logic:
    # 1. Clean up any trailing whitespace.
    # 2. Split the string into words.
    # 3. Take the last word, which should be the house name.
    # 4. Remove any non-alphabetic characters (like a colon or a period).
    sorting_result = ''.join(filter(str.isalpha, generated_text.strip().split()[-1]))

    # Return the formatted result string
    return f"Ah, I see... you belong in... **{sorting_result.upper()}**!"


# --- 3. UI: Create and Launch the Gradio Interface ---
# This part defines the user interface with input fields and a button.
gr_interface = gr.Interface(
    fn=determine_house,
    inputs=[
        gr.Textbox(label="1. What do you value most in a friend?", lines=2),
        gr.Textbox(label="2. If faced with a difficult choice, do you follow your head or your heart?", lines=2),
        gr.Textbox(label="3. What kind of challenges excite you the most?", lines=2),
        gr.Textbox(label="4. How do you react when you fail at something?", lines=2),
        gr.Textbox(label="5. What is your greatest fear?", lines=2)
    ],
    outputs=gr.Markdown(),
    title="The Hogwarts Sorting Hat",
    description="<h3 style='text-align: center;'>Welcome, young witch or wizard! Answer these questions to discover your true Hogwarts house.</h3>",
)

if __name__ == "__main__":
    gr_interface.launch()