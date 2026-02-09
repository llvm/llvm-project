# RUN: env PYTHONPATH=%llvm_lib_dir %python %s %S/../Inputs/input.ll %ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json | FileCheck %s

import sys
import ir2vec

ll_file = sys.argv[1]
vocab_path = sys.argv[2]

# Success case
tool = ir2vec.initEmbedding(filename=ll_file, mode="sym", vocabPath=vocab_path)
print(f"SUCCESS: {type(tool).__name__}")
# CHECK: SUCCESS: IR2VecTool

# Error: Invalid mode
try:
    ir2vec.initEmbedding(filename=ll_file, mode="invalid", vocabPath=vocab_path)
except ValueError:
    print("ERROR: Invalid mode")
# CHECK: ERROR: Invalid mode

# Error: Empty vocab path
try:
    ir2vec.initEmbedding(filename=ll_file, mode="sym", vocabPath="")
except ValueError:
    print("ERROR: Empty vocab path")
# CHECK: ERROR: Empty vocab path

# Error: Invalid file
try:
    ir2vec.initEmbedding(filename="/bad.ll", mode="sym", vocabPath=vocab_path)
except ValueError:
    print("ERROR: Invalid file")
# CHECK: ERROR: Invalid file

# Error: Invalid vocab file
try:
    ir2vec.initEmbedding(filename=ll_file, mode="sym", vocabPath="/bad.json")
except ValueError:
    print("ERROR: Invalid vocab")
# CHECK: ERROR: Invalid vocab

# Error: Malformed JSON vocab
import tempfile
import os
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    f.write("{ this is not valid json }")
    bad_vocab = f.name
try:
    ir2vec.initEmbedding(filename=ll_file, mode="sym", vocabPath=bad_vocab)
except ValueError:
    print("ERROR: Invalid vocab file")
finally:
    os.unlink(bad_vocab)
# CHECK: ERROR: Invalid vocab file