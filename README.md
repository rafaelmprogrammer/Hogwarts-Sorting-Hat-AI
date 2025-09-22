# Hogwarts-Sorting-Hat-AI


✨🧙‍♂️ **The Hogwarts Sorting Hat** 🔮✨

Welcome to your very own Harry Potter experience! This application simulates the magic of the Sorting Hat, using an AI model to determine which Hogwarts house you belong to. Answer five thoughtfully crafted questions and get ready to be sorted into Gryffindor, Hufflepuff, Ravenclaw, or Slytherin.

✨ **Features**

**AI-Powered Sorting:** It utilizes the deepseek-coder-1.3b-instruct model to analyze your responses and make a decision.

**Intuitive Interface:** A simple and user-friendly interface, built with the Gradio library, guides you through the sorting process.

**Magical Results:** Receive your Hogwarts house in an engaging and immersive way.

<hr>

🛠️ **How to Use**


📝 Prerequisites

Ensure you have Python 3.8 or a newer version installed. This application requires a few libraries, which can be installed using pip.

📥 Installation

💻 Clone this repository to your computer:

git clone [https://github.com/o_seu_nome_de_utilizador/nome_do_repositorio.git](https://github.com/o_seu_nome_de_utilizador/nome_do_repositorio.git)

cd nome_do_repositorio


📦 Install the required libraries. You can do this using the provided requirements.txt file, or by installing them one by one:

pip install -r requirements.txt

or

Alternatively, install manually:

pip install torch transformers gradio


🚀 Running the Application

▶️ Execute the app.py script from your terminal:

python app.py


⏱️ Wait while the AI model is downloaded and initialized. The first time you run this, it may take a few minutes.

🌐 Once ready, a URL will be provided in the terminal (typically http://127.0.0.1:7860). Open this URL in your preferred web browser.

🧙‍♀️ Answer the five questions on the interface and click the Submit button to discover your house.

<hr>

🧠 **AI Model Details**

This application uses the deepseek-coder-1.3b-instruct model from DeepSeek AI. This is a compact language model designed to follow instructions, making it perfect for this task. It analyzes the tone and content of your answers to infer your house, just like the Sorting Hat!

<hr>

🙏 **Contributions**

Contributions are always welcome! If you would like to help improve this project, you can:

🐞 **Report Bugs:** If you find an issue, please open a new issue on GitHub.

💡 **Suggest New Features:** Have an idea to make the application even more magical? Share it by opening a new issue.

✍️ **Improve the Code:** Feel free to fork the repository, make your changes, and submit a pull request.

<hr>

📈 **Areas for Improvement**

While the application is functional, there are several areas that could be enhanced in future iterations:

**Model Consistency:** The AI model may occasionally generate unexpected responses that are not house names. For a more consistent output, a fine-tuned model or more robust validation logic would be required.

**Improved Front-end:** The current interface, built with Gradio, is simple and functional. In the future, a custom front-end (e.g., using HTML, CSS, and JavaScript) could offer a more polished design and a richer user experience.

**Improvement of settings**

<p align="center">
Note: This application was created as part of an Ironhack bootcamp, with the assistance of AI tools.
</p>


