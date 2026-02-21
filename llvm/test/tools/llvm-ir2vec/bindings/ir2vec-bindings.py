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

    # Test getInstEmbMap
    print("\n=== Instruction Embeddings ===")

    # Test valid function names in sorted order
    for func_name in sorted(["add", "multiply", "conditional"]):
        inst_emb_map = tool.getInstEmbMap(func_name)
        print(f"Function: {func_name}")
        for inst_str in sorted(inst_emb_map.keys()):
            emb = inst_emb_map[inst_str]
            print(f"  Inst: {inst_str}")
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
# CHECK: === Instruction Embeddings ===
# CHECK: Function: add
# CHECK:   Inst: %sum = add i32 %a, %b
# CHECK-NEXT:     Embedding: [37.0, 38.0, 39.0]
# CHECK:   Inst: ret i32 %sum
# CHECK-NEXT:     Embedding: [1.0, 2.0, 3.0]
# CHECK: Function: conditional
# CHECK:   Inst: %cmp = icmp sgt i32 %n, 0
# CHECK-NEXT:     Embedding: [157.20000000298023, 158.20000000298023, 159.20000000298023]
# CHECK:   Inst: %neg_val = sub i32 %n, 10
# CHECK-NEXT:     Embedding: [43.0, 44.0, 45.0]
# CHECK:   Inst: %pos_val = add i32 %n, 10
# CHECK-NEXT:     Embedding: [37.0, 38.0, 39.0]
# CHECK:   Inst: %result = phi i32 [ %pos_val, %positive ], [ %neg_val, %negative ]
# CHECK-NEXT:     Embedding: [163.0, 164.0, 165.0]
# CHECK:   Inst: br i1 %cmp, label %positive, label %negative
# CHECK-NEXT:     Embedding: [4.0, 5.0, 6.0]
# CHECK:   Inst: br label %exit
# CHECK-NEXT:     Embedding: [4.0, 5.0, 6.0]
# CHECK:   Inst: ret i32 %result
# CHECK-NEXT:     Embedding: [1.0, 2.0, 3.0]
# CHECK: Function: multiply
# CHECK:   Inst: %prod = mul i32 %x, %y
# CHECK-NEXT:     Embedding: [49.0, 50.0, 51.0]
# CHECK:   Inst: ret i32 %prod
# CHECK-NEXT:     Embedding: [1.0, 2.0, 3.0]
