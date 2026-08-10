# Building and installing Python bindings

Python bindings are available as wheels on [PyPI](https://pypi.org/project/google-benchmark/) for importing and 
using Google Benchmark directly in Python. 
Currently, pre-built wheels exist for macOS (both ARM64 and Intel x86), Linux x86-64 and 64-bit Windows.
Supported Python versions are Python 3.8 - 3.12.

To install Google Benchmark's Python bindings, run:

```bash
python -m pip install --upgrade pip  # for manylinux2014 support
python -m pip install google-benchmark
```

In order to keep your system Python interpreter clean, it is advisable to run these commands in a virtual
environment. See the [official Python documentation](https://docs.python.org/3/library/venv.html) 
on how to create virtual environments.

To build a wheel directly from source, you can follow these steps:
```bash
git clone https://github.com/google/benchmark.git
cd benchmark
# create a virtual environment and activate it
python3 -m venv venv --system-site-packages
source venv/bin/activate  # .\venv\Scripts\Activate.ps1 on Windows

# upgrade Python's system-wide packages
python -m pip install --upgrade pip build
# builds the wheel and stores it in the directory "dist".
python -m build
```

NB: Building wheels from source requires Bazel. For platform-specific instructions on how to install Bazel,
refer to the [Bazel installation docs](https://bazel.build/install).
