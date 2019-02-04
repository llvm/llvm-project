# RUN: not llvm-mc -triple=wasm32-unknown-unknown -mattr=+simd128,+nontrapping-fptoint,+exception-handling < %s 2>&1 | FileCheck %s

    .text
    .section    .text.main,"",@
    .type       test0,@function
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
    loop
# CHECK: Block construct type mismatch, expected: end_loop, instead got: end_function
# CHECK: error: Unmatched block construct(s) at function end: loop
# CHECK: error: Unmatched block construct(s) at function end: try
# CHECK: error: Unmatched block construct(s) at function end: block
# CHECK: error: Unmatched block construct(s) at function end: function
    end_function
.Lfunc_end0:
    .size       test0, .Lfunc_end0-test0

