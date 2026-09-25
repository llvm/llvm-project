# RUN: llvm-mc -filetype=obj -triple=wasm32-unknown-unknown %s -o %t.o
# RUN: obj2yaml %t.o | FileCheck %s

        .comm x,4,4
        .comm y,8,8

        .hidden z
        .comm z,16,16

# CHECK:        - Type:            CUSTOM
# CHECK-NEXT:     Name:            linking
# CHECK-NEXT:     Version:         2
# CHECK-NEXT:     SymbolTable:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Kind:            DATA
# CHECK-NEXT:         Name:            x
# CHECK-NEXT:         Flags:           [ BINDING_COMMON ]
# CHECK-NEXT:         Size:            4
# CHECK-NEXT:         Align:           4
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Kind:            DATA
# CHECK-NEXT:         Name:            y
# CHECK-NEXT:         Flags:           [ BINDING_COMMON ]
# CHECK-NEXT:         Size:            8
# CHECK-NEXT:         Align:           8
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Kind:            DATA
# CHECK-NEXT:         Name:            z
# CHECK-NEXT:         Flags:           [ BINDING_COMMON, VISIBILITY_HIDDEN ]
# CHECK-NEXT:         Size:            16
# CHECK-NEXT:         Align:           16
