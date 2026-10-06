# Build Instructions

## Setup

```bash
pip install -r requirements.txt
```

## Building the Documentation Locally

Inside of the activated virtual environment, run the following command to build the documentation:

```bash
python -m sphinx -T -b html -d _build/doctrees -D language=en . test_build/html
```

After which we can find the built documentation in `test_build/html`.
