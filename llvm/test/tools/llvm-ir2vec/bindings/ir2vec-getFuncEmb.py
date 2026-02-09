# RUN: env PYTHONPATH=%llvm_lib_dir %python %s %S/../Inputs/input.ll %ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json | FileCheck %s

import sys
import ir2vec

ll_file = sys.argv[1]
vocab_path = sys.argv[2]

tool = ir2vec.initEmbedding(filename=ll_file, mode="sym", vocabPath=vocab_path)

# Success case
emb = tool.getFuncEmb("add")
print(f"SUCCESS: {emb.tolist()}")
# CHECK: SUCCESS: [38.0, 40.0, 42.0]

# Error: Function not found
try:
    tool.getFuncEmb("nonexistent")
except ValueError:
    print("ERROR: Function not found")
# CHECK: ERROR: Function not found