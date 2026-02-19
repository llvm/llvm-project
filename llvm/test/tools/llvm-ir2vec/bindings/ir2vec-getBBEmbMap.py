# RUN: env PYTHONPATH=%llvm_lib_dir %python %s %S/../Inputs/input.ll %ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json | FileCheck %s

import sys
import ir2vec

ll_file = sys.argv[1]
vocab_path = sys.argv[2]

tool = ir2vec.initEmbedding(filename=ll_file, mode="sym", vocabPath=vocab_path)

# Success case
bb_map = tool.getBBEmbMap("conditional")
for bb in sorted(bb_map.keys()):
    print(f"BB: {bb}, EMB: {bb_map[bb].tolist()}")
# CHECK: BB: entry, EMB: [161.20000000298023, 163.20000000298023, 165.20000000298023]
# CHECK: BB: exit, EMB: [164.0, 166.0, 168.0]
# CHECK: BB: negative, EMB: [47.0, 49.0, 51.0]
# CHECK: BB: positive, EMB: [41.0, 43.0, 45.0]

# Error: Function not found
try:
    tool.getBBEmbMap("nonexistent")
except ValueError:
    print("ERROR: Function not found")
# CHECK: ERROR: Function not found