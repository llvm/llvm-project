# RUN: env PYTHONPATH=%llvm_lib_dir %python %s %S/../Inputs/input.ll %ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json | FileCheck %s

import sys
import ir2vec
import tempfile
import os

ll_file = sys.argv[1]
vocab_path = sys.argv[2]

# ============================================================
# loadVocab tests
# ============================================================

# Success: Load a valid vocabulary
vocab = ir2vec.loadVocab(vocab_path)
print(f"VOCAB: {type(vocab).__name__}")
# CHECK: VOCAB: Vocab

# Error: Empty vocab path
try:
    ir2vec.loadVocab("")
except ValueError:
    print("ERROR: Empty vocab path")
# CHECK: ERROR: Empty vocab path

# Error: Non-existent vocab file
try:
    ir2vec.loadVocab("/nonexistent/path/bad.json")
except ValueError:
    print("ERROR: Invalid vocab path")
# CHECK: ERROR: Invalid vocab path

# Error: Malformed JSON vocab file
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    f.write("{ this is not valid json }")
    bad_vocab = f.name
try:
    ir2vec.loadVocab(bad_vocab)
except ValueError:
    print("ERROR: Malformed vocab file")
finally:
    os.unlink(bad_vocab)
# CHECK: ERROR: Malformed vocab file

# Error: Wrong type for vocab path (not a string)
try:
    ir2vec.loadVocab(42)
except TypeError:
    print("ERROR: Invalid vocab path type")
# CHECK: ERROR: Invalid vocab path type

# ============================================================
# initEmbedding tests
# ============================================================

# Success: Create embedding tool with valid inputs
tool = ir2vec.initEmbedding(
    filename=ll_file, mode=ir2vec.IR2VecKind.Symbolic, vocab=vocab
)
print(f"SUCCESS: {type(tool).__name__}")
# CHECK: SUCCESS: IR2VecTool

# Error: Invalid mode (string instead of IR2VecKind enum)
try:
    ir2vec.initEmbedding(filename=ll_file, mode="invalid", vocab=vocab)
except TypeError:
    print("ERROR: Invalid mode type")
# CHECK: ERROR: Invalid mode type

# Error: Invalid mode (integer instead of IR2VecKind enum)
try:
    ir2vec.initEmbedding(filename=ll_file, mode=99, vocab=vocab)
except TypeError:
    print("ERROR: Invalid mode int")
# CHECK: ERROR: Invalid mode int

# Error: Non-existent IR file
try:
    ir2vec.initEmbedding(
        filename="/nonexistent/bad.ll",
        mode=ir2vec.IR2VecKind.Symbolic,
        vocab=vocab,
    )
except ValueError:
    print("ERROR: Invalid file")
# CHECK: ERROR: Invalid file

# Error: Empty filename
try:
    ir2vec.initEmbedding(filename="", mode=ir2vec.IR2VecKind.Symbolic, vocab=vocab)
except ValueError:
    print("ERROR: Empty filename")
# CHECK: ERROR: Empty filename

# Error: Wrong type for vocab (string instead of Vocab object)
try:
    ir2vec.initEmbedding(
        filename=ll_file, mode=ir2vec.IR2VecKind.Symbolic, vocab="vocab.json"
    )
except TypeError:
    print("ERROR: Vocab is string")
# CHECK: ERROR: Vocab is string

# Error: Wrong type for vocab (None)
try:
    ir2vec.initEmbedding(filename=ll_file, mode=ir2vec.IR2VecKind.Symbolic, vocab=None)
except TypeError:
    print("ERROR: Vocab is None")
# CHECK: ERROR: Vocab is None

# Error: Wrong type for vocab (integer)
try:
    ir2vec.initEmbedding(filename=ll_file, mode=ir2vec.IR2VecKind.Symbolic, vocab=123)
except TypeError:
    print("ERROR: Vocab is int")
# CHECK: ERROR: Vocab is int

# Error: Missing vocab argument entirely
try:
    ir2vec.initEmbedding(filename=ll_file, mode=ir2vec.IR2VecKind.Symbolic)
except TypeError:
    print("ERROR: Vocab missing")
# CHECK: ERROR: Vocab missing

# Error: Wrong type for filename (not a string)
try:
    ir2vec.initEmbedding(filename=42, mode=ir2vec.IR2VecKind.Symbolic, vocab=vocab)
except TypeError:
    print("ERROR: Filename is int")
# CHECK: ERROR: Filename is int
