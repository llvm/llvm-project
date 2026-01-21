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
    func_emb_map = tool.getFuncEmbMap()
    print(f"Number of functions: {len(func_emb_map)}")

    # Check that all three functions are present
    expected_funcs = ["add", "multiply", "conditional"]
    for func_name in expected_funcs:
        if func_name in func_emb_map:
            emb = func_emb_map[func_name]
            print(f"Function '{func_name}': embedding shape = {emb.shape}")
        else:
            print(f"ERROR: Function '{func_name}' not found")

# CHECK: SUCCESS: Tool initialized
# CHECK: Tool type: IR2VecTool
# CHECK: Number of functions: 3
# CHECK: Function 'add': embedding shape =
# CHECK: Function 'multiply': embedding shape =
# CHECK: Function 'conditional': embedding shape =
