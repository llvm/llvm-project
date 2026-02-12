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

    # Test getBBEmbMap
    print("\n=== Basic Block Embeddings ===")

    # Test valid function names in sorted order
    for func_name in sorted(["add", "multiply", "conditional"]):
        bb_emb_map = tool.getBBEmbMap(func_name)
        print(f"Function: {func_name}")
        for bb_name in sorted(bb_emb_map.keys()):
            emb = bb_emb_map[bb_name]
            print(f"  BB: {bb_name}")
            print(f"    Embedding: {emb.tolist()}")

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
# CHECK: === Basic Block Embeddings ===
# CHECK: Function: add
# CHECK:   BB: entry
# CHECK-NEXT:     Embedding: [38.0, 40.0, 42.0]
# CHECK: Function: conditional
# CHECK:   BB: entry
# CHECK-NEXT:     Embedding: [161.20000000298023, 163.20000000298023, 165.20000000298023]
# CHECK:   BB: exit
# CHECK-NEXT:     Embedding: [164.0, 166.0, 168.0]
# CHECK:   BB: negative
# CHECK-NEXT:     Embedding: [47.0, 49.0, 51.0]
# CHECK:   BB: positive
# CHECK-NEXT:     Embedding: [41.0, 43.0, 45.0]
# CHECK: Function: multiply
# CHECK:   BB: entry
# CHECK-NEXT:     Embedding: [50.0, 52.0, 54.0]
