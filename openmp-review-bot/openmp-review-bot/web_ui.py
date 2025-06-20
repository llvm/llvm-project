import gradio as gr
from llama_cpp import Llama

# Load the model
llm = Llama(model_path="models/tinyllama.gguf", n_ctx=2048)

def review_openmp_code(code, optimization_focus, detail_level):
    prompt = (
        f"You are an advanced AI code reviewer. Analyze the following OpenMP C/C++ code.\n"
        f"Focus on: {optimization_focus}.\n"
        f"Detail level: {detail_level}.\n"
        f"\nCode:\n{code}\n\nDetailed Review:"
    )
    response = llm(prompt=prompt, max_tokens=512, stop=["</s>"])
    return response["choices"][0]["text"].strip()

def summarize_code(code):
    prompt = f"Summarize the following OpenMP C/C++ code in simple terms:\n\n{code}\n\nSummary:"
    response = llm(prompt=prompt, max_tokens=256, stop=["</s>"])
    return response["choices"][0]["text"].strip()

with gr.Blocks(title="🔍 OpenMP Code Reviewer AI") as demo:
    gr.HTML("""
    <style>
        #review-btn { background-color: #4CAF50; color: white; font-size: 16px; padding: 12px 24px; border-radius: 8px; margin-top: 10px; }
        #summary-btn { background-color: #2196F3; color: white; font-size: 16px; padding: 12px 24px; border-radius: 8px; margin-top: 10px; }

        /* Welcome overlay styles */
        #welcome-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: white;
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            font-weight: bold;
            color: #4CAF50;
            animation: fadeOut 1s ease-in-out 4s forwards;
        }

        @keyframes fadeOut {
            to {
                opacity: 0;
                visibility: hidden;
            }
        }
    </style>

    <div id="welcome-overlay">👋 Welcome to OpenMP AI Code Reviewer</div>

    <h1 style='text-align: center; color: #4CAF50;'>🧠 OpenMP AI Code Reviewer</h1>
    <p style='text-align: center;'>Paste or upload your OpenMP C/C++ code for detailed analysis and simplified summaries.</p>
    """)

    with gr.Row():
        code_input = gr.Code(label="📝 Input C/C++ Code", language="cpp", lines=20)
        file_input = gr.File(label="📁 Upload C/C++ File", file_types=[".cpp", ".c"])

    with gr.Row():
        optimization_focus = gr.Dropdown(
            choices=["Race Conditions", "Parallel Efficiency", "Memory Usage", "Synchronization", "All"],
            label="🔧 Optimization Focus",
            value="All"
        )
        detail_level = gr.Radio(
            choices=["High", "Medium", "Low"],
            label="📊 Detail Level",
            value="High"
        )

    with gr.Row():
        review_btn = gr.Button("🚀 Run Review", elem_id="review-btn")
        summary_btn = gr.Button("📘 Generate Summary", elem_id="summary-btn")

    with gr.Tab("🔎 AI Review"):
        review_output = gr.Textbox(label="📋 Detailed Review", lines=20, interactive=False, show_copy_button=True)

    with gr.Tab("📄 Code Summary"):
        summary_output = gr.Textbox(label="🧾 Simplified Summary", lines=10, interactive=False, show_copy_button=True)

    review_btn.click(fn=review_openmp_code, inputs=[code_input, optimization_focus, detail_level], outputs=review_output)
    summary_btn.click(fn=summarize_code, inputs=code_input, outputs=summary_output)

    def load_code(file):
        with open(file.name, 'r') as f:
            return f.read()

    file_input.change(fn=load_code, inputs=file_input, outputs=code_input)

    gr.HTML("<footer style='text-align: center; padding-top: 20px;'>Made with ❤️ using TinyLlama + Gradio</footer>")

# Launch the app
demo.launch(share=True)
