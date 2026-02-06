# RUN: env PYTHONPATH=%llvm_lib_dir %python %s %S/../Inputs/input.ll %ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json | FileCheck %s

import sys
import ir2vec

ll_file = sys.argv[1]
vocab_path = sys.argv[2]

tool = ir2vec.initEmbedding(filename=ll_file, mode="sym", vocabPath=vocab_path)

if tool is not None:
    print("SUCCESS: Tool initialized")
    print(f"Tool type: {type(tool).__name__}")

    # Test getFuncEmbMap
    print("\n=== Function Embeddings ===")
    func_emb_map = tool.getFuncEmbMap()

    # Sorting the function names for fixed-ordered output
    for func_name in sorted(func_emb_map.keys()):
        emb = func_emb_map[func_name]
        print(f"Function: {func_name}")
        print(f"  Embedding: {emb.tolist()}")

    # Test getFuncEmb for individual functions
    print("\n=== Single Function Embeddings ===")

    # Test valid function names
    for func_name in ["add", "multiply", "conditional"]:
        func_emb = tool.getFuncEmb(func_name)
        print(f"Function: {func_name}")
        print(f"  Embedding: {func_emb.tolist()}")

# CHECK: SUCCESS: Tool initialized
# CHECK: Tool type: IR2VecTool
# CHECK: === Function Embeddings ===
# CHECK: Function: add
# CHECK-NEXT:   Embedding: [38.0, 40.0, 42.0]
# CHECK: Function: conditional
# CHECK-NEXT:   Embedding: [413.20000000298023, 421.20000000298023, 429.20000000298023]
# CHECK: Function: multiply
# CHECK-NEXT:   Embedding: [50.0, 52.0, 54.0]
# CHECK: === Single Function Embeddings ===
# CHECK: Function: add
# CHECK-NEXT:   Embedding: [38.0, 40.0, 42.0]
# CHECK: Function: multiply
# CHECK-NEXT:   Embedding: [50.0, 52.0, 54.0]
# CHECK: Function: conditional
# CHECK-NEXT:   Embedding: [413.20000000298023, 421.20000000298023, 429.20000000298023]
