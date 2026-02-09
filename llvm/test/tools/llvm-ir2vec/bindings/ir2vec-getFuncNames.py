# RUN: env PYTHONPATH=%llvm_lib_dir %python %s %S/../Inputs/input.ll %ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json | FileCheck %s

import sys
import ir2vec

ll_file = sys.argv[1]
vocab_path = sys.argv[2]

tool = ir2vec.initEmbedding(filename=ll_file, mode="sym", vocabPath=vocab_path)

# Success case
func_names = tool.getFuncNames()
for name in sorted(func_names):
    print(f"FUNC: {name}")

# CHECK: FUNC: add
# CHECK: FUNC: conditional
# CHECK: FUNC: multiply