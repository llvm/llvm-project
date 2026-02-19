# RUN: env PYTHONPATH=%llvm_lib_dir %python %s %S/../Inputs/input.ll %ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json | FileCheck %s

import sys
import ir2vec

ll_file = sys.argv[1]
vocab_path = sys.argv[2]

tool = ir2vec.initEmbedding(filename=ll_file, mode="sym", vocabPath=vocab_path)

# Success case
emb_map = tool.getFuncEmbMap()
for name in sorted(emb_map.keys()):
    print(f"FUNC: {name}, EMB: {emb_map[name].tolist()}")

# CHECK: FUNC: add, EMB: [38.0, 40.0, 42.0]
# CHECK: FUNC: conditional, EMB: [413.20000000298023, 421.20000000298023, 429.20000000298023]
# CHECK: FUNC: multiply, EMB: [50.0, 52.0, 54.0]