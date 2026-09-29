# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj -o %t.o -mattr=+simd128,+nontrapping-fptoint,+exception-handling < %s
# RUN: llvm-objdump --triple=wasm32-unknown-unknown -d %t.o | FileCheck %s

test0:
    .functype   test0 (i32, i64) -> (f32)
    .local      f32, f64, v128, v128
    local.get   2
    end_function

test1:
    .functype   test1 (i32, i64) -> (i64)
    .local      i32, i64, funcref
    local.get   3
    end_function


# CHECK-LABEL: CODE:
# CHECK:             # 2 functions in section.
# CHECK-LABEL: <test0>:
# CHECK-NEXT:        .local  f32, f64, v128, v128
# CHECK-NEXT:       9:       20 02  local.get	2
# CHECK-NEXT:       b:       0b     end
# CHECK-LABEL: <test1>:
# CHECK-NEXT:        .local  i32, i64, funcref
# CHECK-NEXT:      14:       20 03  local.get	3
# CHECK-NEXT:      16:       0b     end
