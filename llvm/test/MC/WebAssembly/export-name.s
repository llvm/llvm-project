# RUN: llvm-mc -triple=wasm32-unknown-unknown < %s | FileCheck %s
# Check that it also comiled to object for format.
# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj -o - < %s | obj2yaml | FileCheck -check-prefix=CHECK-OBJ %s

foo:
    .globl foo
    .functype foo () -> ()
    .export_name foo, bar
    end_function

square:
    .globl square
    .functype square () -> ()
    .export_name square, "[square]"
    end_function

mid$dollar:
    .globl mid$dollar
    .functype mid$dollar () -> ()
    .export_name mid$dollar, mid$dollar
    end_function

mid?question:
    .globl mid?question
    .functype mid?question () -> ()
    .export_name mid?question, mid?question
    end_function

# CHECK: .export_name foo, "bar"
# CHECK: .export_name square, "[square]"
# CHECK: .export_name mid$dollar, "mid$dollar"
# CHECK: .export_name mid?question, "mid?question"

# CHECK-OBJ:        - Type:            EXPORT
# CHECK-OBJ-NEXT:     Exports:
# CHECK-OBJ-NEXT:       - Name:            bar
# CHECK-OBJ-NEXT:         Kind:            FUNCTION
# CHECK-OBJ-NEXT:         Index:           0
# CHECK-OBJ-NEXT:       - Name:            '[square]'
# CHECK-OBJ-NEXT:         Kind:            FUNCTION
# CHECK-OBJ-NEXT:         Index:           1
# CHECK-OBJ-NEXT:       - Name:            'mid$dollar'
# CHECK-OBJ-NEXT:         Kind:            FUNCTION
# CHECK-OBJ-NEXT:         Index:           2
# CHECK-OBJ-NEXT:       - Name:            'mid?question'
# CHECK-OBJ-NEXT:         Kind:            FUNCTION
# CHECK-OBJ-NEXT:         Index:           3

# CHECK-OBJ:          Name:            linking
# CHECK-OBJ-NEXT:     Version:         2
# CHECK-OBJ-NEXT:     SymbolTable:
# CHECK-OBJ-NEXT:       - Index:           0
# CHECK-OBJ-NEXT:         Kind:            FUNCTION
# CHECK-OBJ-NEXT:         Name:            foo
# CHECK-OBJ-NEXT:         Flags:           [ EXPORTED ]
# CHECK-OBJ-NEXT:         Function:        0
# CHECK-OBJ-NEXT:       - Index:           1
# CHECK-OBJ-NEXT:         Kind:            FUNCTION
# CHECK-OBJ-NEXT:         Name:            square
# CHECK-OBJ-NEXT:         Flags:           [ EXPORTED ]
# CHECK-OBJ-NEXT:         Function:        1
# CHECK-OBJ-NEXT:       - Index:           2
# CHECK-OBJ-NEXT:         Kind:            FUNCTION
# CHECK-OBJ-NEXT:         Name:            'mid$dollar'
# CHECK-OBJ-NEXT:         Flags:           [ EXPORTED ]
# CHECK-OBJ-NEXT:         Function:        2
# CHECK-OBJ-NEXT:       - Index:           3
# CHECK-OBJ-NEXT:         Kind:            FUNCTION
# CHECK-OBJ-NEXT:         Name:            'mid?question'
# CHECK-OBJ-NEXT:         Flags:           [ EXPORTED ]
# CHECK-OBJ-NEXT:         Function:        3
