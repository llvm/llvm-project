from llama_cpp import Llama

llm = Llama(model_path=r"C:\Users\bhumi\openmp-review-bot\models\tinyllama.gguf", n_ctx=2048)

response = llm.create_completion(
    prompt="### Human: What is OpenMP?\n### Assistant:",
    max_tokens=200,
    stop=["### Human:"]
)

print(response['choices'][0]['text'].strip())
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    prompt = f"### Human: {user_input}\n### Assistant:"
    response = llm.create_completion(
        prompt=prompt,
        max_tokens=300,
        stop=["### Human:"]
    )
    print("AI:", response['choices'][0]['text'].strip())
