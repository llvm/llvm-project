# RUN: env PYTHONPATH=%llvm_lib_dir %python %s %S/../Inputs/input.ll %ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json | FileCheck %s

import sys
import ir2vec

ll_file = sys.argv[1]
vocab_path = sys.argv[2]

tool = ir2vec.initEmbedding(filename=ll_file, mode="sym", vocabPath=vocab_path)

if tool is not None:
    print("SUCCESS: Tool initialized")
    print(f"Tool type: {type(tool).__name__}")

# CHECK: SUCCESS: Tool initialized
# CHECK: Tool type: IR2VecTool
