# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj -o %t.o -mattr=+simd128,+nontrapping-fptoint,+exception-handling < %s
# RUN: llvm-objdump -triple=wasm32-unknown-unknown -disassemble %t.o | FileCheck %s

    .section .text.main1,"",@
    .type    test0,@function
test0:
    .functype   test0 (i32, i64) -> (i32)
    .local      f32, f64, v128, v128
    local.get   2
    end_function
.Lfunc_end0:
    .size	test0, .Lfunc_end0-test0

    .section .text.main2,"",@
    .type    test1,@function
test1:
    .functype   test1 (i32, i64) -> (i32)
    .local      i32, i64, except_ref
    local.get   3
    end_function
.Lfunc_end1:
    .size	test1, .Lfunc_end1-test1


# CHECK-LABEL: CODE:
# CHECK:             # 2 functions in section.
# CHECK-LABEL: test0:
# CHECK-NEXT:        .local  f32, f64, v128, v128
# CHECK-NEXT:       9:       20 02  local.get	2
# CHECK-NEXT:       b:       0b     end_block
# CHECK-LABEL: test1:
# CHECK-NEXT:        .local  i32, i64, except_ref
# CHECK-NEXT:      14:       20 03  local.get	3
# CHECK-NEXT:      16:       0b     end_block
