# Test that placing multiple data symbols in the same section works

# RUN: llvm-mc -triple=wasm32-unknown-unknown < %s | FileCheck %s

test0:
    .functype   test0 () -> (i32, i32)
    i32.const a
    i32.const b
    end_function

    .section mysec,"",@
a:
    .int32 42
    .int32 43
    .size a, 8
b:
    .int32 44
    .size b, 4

#      CHECK:   .section  mysec,"",@
# CHECK-NEXT: a:
# CHECK-NEXT:   .int32 42
# CHECK-NEXT:   .int32 43
# CHECK-NEXT:   .size a, 8
# CHECK-NEXT: b:
# CHECK-NEXT:   .int32 44
# CHECK-NEXT:   .size b, 4

# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown < %s | llvm-objdump --triple=wasm32-unknown-unknown -d -t -r - | FileCheck %s --check-prefix=OBJ


#      OBJ: 00000001 <test0>:
#      OBJ:        3: 41 80 80 80 80 00     i32.const       0
# OBJ-NEXT:                         00000004:  R_WASM_MEMORY_ADDR_SLEB      a+0
# OBJ-NEXT:        9: 41 88 80 80 80 00     i32.const       8
# OBJ-NEXT:                         0000000a:  R_WASM_MEMORY_ADDR_SLEB      b+0
# OBJ-NEXT:        f: 0b            end
