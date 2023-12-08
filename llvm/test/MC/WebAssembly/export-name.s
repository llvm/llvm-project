# RUN: llvm-mc -triple=wasm32-unknown-unknown < %s | FileCheck %s
# Check that it also comiled to object for format.
# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj -o - < %s | obj2yaml | FileCheck -check-prefix=CHECK-OBJ %s

foo:
    .globl foo
    .functype foo () -> ()
    .export_name foo, bar
    end_function

# CHECK: .export_name foo, bar

# CHECK-OBJ:        - Type:            EXPORT
# CHECK-OBJ-NEXT:     Exports:
# CHECK-OBJ-NEXT:       - Name:            bar
# CHECK-OBJ-NEXT:         Kind:            FUNCTION
# CHECK-OBJ-NEXT:         Index:           0

# CHECK-OBJ:          Name:            linking
# CHECK-OBJ-NEXT:     Version:         2
# CHECK-OBJ-NEXT:     SymbolTable:
# CHECK-OBJ-NEXT:       - Index:           0
# CHECK-OBJ-NEXT:         Kind:            FUNCTION
# CHECK-OBJ-NEXT:         Name:            foo
# CHECK-OBJ-NEXT:         Flags:           [ EXPORTED ]
# CHECK-OBJ-NEXT:         Function:        0
