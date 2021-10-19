# RUN: not llvm-mc -triple=wasm32-unknown-unknown -mattr=+simd128,+nontrapping-fptoint,+exception-handling < %s 2>&1 | FileCheck %s

# CHECK: invalid operand for instruction
# (must be 0.0 or similar)
    f32.const 0

# CHECK: Wasm doesn't support data symbols in text sections
	.type	objerr,@object
objerr:

# CHECK: End of block construct with no start: end_try
    end_try
test0:
    .functype   test0 () -> ()
# CHECK: Block construct type mismatch, expected: end_function, instead got: end_loop
    end_loop
    block
# CHECK: Block construct type mismatch, expected: end_block, instead got: end_if
    end_if
    try
# CHECK: Block construct type mismatch, expected: end_try/delegate, instead got: end_block
    end_block
    loop
    try
    catch_all
    catch_all
# CHECK: error: Block construct type mismatch, expected: end_try, instead got: catch_all
    end
# CHECK: Block construct type mismatch, expected: end_try, instead got: end_function
# CHECK: error: Unmatched block construct(s) at function end: catch_all
# CHECK: error: Unmatched block construct(s) at function end: loop
# CHECK: error: Unmatched block construct(s) at function end: try
# CHECK: error: Unmatched block construct(s) at function end: block
# CHECK: error: Unmatched block construct(s) at function end: function
    end_function

