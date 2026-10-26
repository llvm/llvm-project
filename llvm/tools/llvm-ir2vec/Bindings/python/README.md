# IR2Vec Python Bindings

Python bindings for [IR2Vec](https://llvm.org/docs/MLGO.html#ir2vec), an LLVM
IR embedding framework that generates vector representations of LLVM IR for use
in machine learning-based compiler optimization.

## Installation

```bash
pip install ir2vec
```

## Usage

```python
import ir2vec

# Symbolic embeddings using a bundled vocabulary
vocabObj = ir2vec.loadVocab(ir2vec.SEED_EMBEDDING_75D)

emb = ir2vec.initEmbedding (
    filename="file_path.ll",
    mode=ir2vec.IR2VecKind.FlowAware,
    vocab=vocabObj
)


func_names = emb.getFuncNames()
func_emb_map = emb.getFuncEmbMap()

# for an IR function "foo"
func_emb = emb.getFuncEmb("foo")
bb_map = emb.getBBEmbMap("foo")
inst_map = emb.getInstEmbMap("foo")
```

## Bundled vocabularies

The package ships a pre-trained seed embedding vocabulary:

| Attribute | Dimensions |
|---|---|
| `ir2vec.SEED_EMBEDDING_75D` | 75 |

This pre-packaged vocab file has been taken from [`Analysis/models/seedEmbeddingVocab75D.json`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/Analysis/models/seedEmbeddingVocab75D.json) in the LLVM monorepo. Pass this or a vocab of your choice directly to `loadVocab(vocabPath=...)`.

## Building from source

The bindings can be built as part of the LLVM monorepo. From the repo root:

```bash
cmake -G Ninja -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_IR2VEC_ENABLE_PYTHON_BINDINGS=ON \
  llvm

ninja -C build llvm-ir2vec
```

This produces the native extension and a self-contained Python package under
`build/tools/llvm-ir2vec/Bindings/python/`, with the vocabulary file already in place.
To build and test a wheel from it:

1. Build the wheel:

    ```bash
    pip wheel build/tools/llvm-ir2vec/Bindings/python/ \
    --no-build-isolation \
    --no-deps \
    --wheel-dir dist/
    ```

2. Install the resulting wheel:

    ```bash
    pip install dist/ir2vec-*.whl
    ```

## License

Apache License v2.0 with LLVM Exceptions. See
[LICENSE.TXT](LICENSE.TXT) or [llvm.org/LICENSE.txt](https://llvm.org/LICENSE.txt).
