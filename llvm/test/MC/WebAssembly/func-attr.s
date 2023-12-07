# RUN: llvm-mc -triple=wasm32-unknown-unknown < %s | FileCheck %s
# Check that it also comiled to object for format.
# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj -o - < %s | obj2yaml | FileCheck -check-prefix=CHECK-OBJ %s

foo:
    .globl foo
    .functype foo () -> ()
    end_function

    .section        .custom_section.llvm.func_attr.custom0,"",@
    .int32  foo@FUNCINDEX

# CHECK:       .section .custom_section.llvm.func_attr.custom0,"",@
# CHECK-NEXT: .int32  foo@FUNCINDEX

# CHECK-OBJ:        - Type:            CUSTOM
# CHECK-OBJ-NEXT:     Relocations:
# CHECK-OBJ-NEXT:        - Type:            R_WASM_FUNCTION_INDEX_I32
# CHECK-OBJ-NEXT:          Index:           0
# CHECK-OBJ-NEXT:          Offset:          0x0
# CHECK-OBJ-NEXT:     Name:            llvm.func_attr.custom0
