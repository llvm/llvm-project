# RUN: env PYTHONPATH=%llvm_lib_dir %python %s %S/../Inputs/input.ll %ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json | FileCheck %s

import sys
import ir2vec

ll_file = sys.argv[1]
vocab_path = sys.argv[2]

tool = ir2vec.initEmbedding(filename=ll_file, mode="sym", vocabPath=vocab_path)

# Success case
inst_map = tool.getInstEmbMap("add")
for inst in sorted(inst_map.keys()):
    print(f"INST: {inst}")
    print(f"  EMB: {inst_map[inst].tolist()}")

# CHECK: INST: %sum = add i32 %a, %b
# CHECK:   EMB: [37.0, 38.0, 39.0]
# CHECK: INST: ret i32 %sum
# CHECK:   EMB: [1.0, 2.0, 3.0]

# Error: Function not found
try:
    tool.getInstEmbMap("nonexistent")
except ValueError:
    print("ERROR: Function not found")
# CHECK: ERROR: Function not found