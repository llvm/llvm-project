# RUN: llvm-mc -triple=wasm32-unknown-unknown < %s | FileCheck %s
# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj < %s | obj2yaml | FileCheck -check-prefix=BIN %s

# Test creating several empty tables

.tabletype foo, externref        
foo:

bar:
.tabletype bar, funcref

# CHECK: .tabletype foo, externref
# CHECK: foo:

#      CHECK: bar:
# CHECK-NEXT: .tabletype bar, funcref

#      BIN:  - Type:            TABLE
# BIN-NEXT:    Tables:
# BIN-NEXT:      - Index:           1
# BIN-NEXT:        ElemType:        EXTERNREF
# BIN-NEXT:        Limits:
# BIN-NEXT:          Initial:         0x00000000
# BIN-NEXT:      - Index:           2
# BIN-NEXT:        ElemType:        FUNCREF
# BIN-NEXT:        Limits:
# BIN-NEXT:          Initial:         0x00000000

#      BIN:  - Type:            CUSTOM
# BIN-NEXT:    Name:            linking
# BIN-NEXT:    Version:         2
# BIN-NEXT:    SymbolTable:
# BIN-NEXT:      - Index:           0
# BIN-NEXT:        Kind:            TABLE
# BIN-NEXT:        Name:            foo
# BIN-NEXT:        Flags:           [ BINDING_LOCAL ]
# BIN-NEXT:        Table:           1
# BIN-NEXT:      - Index:           1
# BIN-NEXT:        Kind:            TABLE
# BIN-NEXT:        Name:            bar
# BIN-NEXT:        Flags:           [ BINDING_LOCAL ]
# BIN-NEXT:        Table:           2
