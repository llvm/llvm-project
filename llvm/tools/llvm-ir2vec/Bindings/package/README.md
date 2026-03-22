<!--
Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# ir2vec

Python bindings for [IR2Vec](https://llvm.org/docs/MLGO.html#ir2vec), an LLVM IR
embedding framework that generates vector representations of LLVM IR for use
in machine learning-based compiler optimization.

## Requirements

- Python >= 3.10
- NumPy
- An IR2Vec vocabulary file (see the IR2Vec documentation)

## Building a Wheel

These bindings require a pre-built shared library. The following steps build a
wheel by packaging the compiled shared library produced during the LLVM build.

**Step 1: Build LLVM with Python bindings enabled**

Configure and build LLVM with the `-DLLVM_IR2VEC_ENABLE_PYTHON_BINDINGS=ON`
flag:
```bash
cmake -G Ninja -S llvm-project/llvm -B build-llvm \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_IR2VEC_ENABLE_PYTHON_BINDINGS=ON \
  -DPython_EXECUTABLE=$(which python3) \
  -DPython3_EXECUTABLE=$(which python3) \
  ...

ninja -C build-llvm
```

**Step 2: Locate the built shared library**
```bash
find build-llvm/ -name "ir2vec*.so"
```

**Step 3: Copy the shared library into the package directory**
```bash
cp build-llvm/lib/ir2vec*.so \
  llvm-project/llvm/tools/llvm-ir2vec/Bindings/package/
```

**Step 4: Build the wheel**
```bash
cd llvm-project/llvm/tools/llvm-ir2vec/Bindings/package/
pip install --upgrade pip setuptools wheel
pip wheel . --no-build-isolation --no-cache-dir -w dist/
```

The wheel will be written to `dist/`.

**Step 5: Install**
```bash
pip install dist/ir2vec-*.whl
```

## Usage
```python
import ir2vec

tool = ir2vec.initEmbedding(
    filename="module.ll",
    mode="sym",
    vocabPath="/path/to/vocab.json"
)
embeddings = tool.getFuncEmbMap()
```

## Source

This package is part of the LLVM Project:
https://github.com/llvm/llvm-project/tree/main/llvm/tools/llvm-ir2vec

## License

Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for details.
