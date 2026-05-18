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
vocabObj = ir2vec.loadVocab(
    ir2vec.vocab.seedEmbedding75D
)

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
| `ir2vec.vocab.seedEmbedding75D` | 75 |

Pass this or a vocab of your choice directly to `loadVocab(vocabPath=...)`.

## Building from source

The bindings are built as part of the LLVM monorepo. From the repo root:

```bash
cmake -G Ninja -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_IR2VEC_ENABLE_PYTHON_BINDINGS=ON \
  llvm

ninja -C build llvm-ir2vec
```

## License

Apache License v2.0 with LLVM Exceptions. See
[llvm.org/LICENSE.txt](https://llvm.org/LICENSE.txt).