# RUN: env PYTHONPATH=%llvm_lib_dir %python %s %S/test.ll %ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json | FileCheck %s

import sys
import py_ir2vec

ll_file = sys.argv[1]
vocab_path = sys.argv[2]

tool = py_ir2vec.initEmbedding(filename=ll_file, mode="sym", vocab_override=vocab_path)

if tool is not None:
    print("SUCCESS: Tool initialized")
    print(f"Tool type: {type(tool).__name__}")

# CHECK: SUCCESS: Tool initialized
# CHECK: Tool type: IR2VecTool
