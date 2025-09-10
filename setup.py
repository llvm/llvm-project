from setuptools import setup

setup(
    name="omp-pr-summary",
    version="0.1",
    py_modules=["CLI_CLANG_TOOL3"],
    install_requires=[
        "typer[all]",
        "requests",
        "faiss-cpu",
        "sentence-transformers",
        "openai",
        "torch",
        "numpy"
    ],
    entry_points={
        "console_scripts": [
            "omp-pr-summary=CLI_CLANG_TOOL3:app",
        ],
    },
)
