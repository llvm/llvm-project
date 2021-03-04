# RUN: llvm-mc -triple=wasm32-unknown-unknown -mattr=+reference-types < %s | FileCheck --check-prefix=CHECK %s
# RUN: llvm-mc -triple=wasm32-unknown-unknown -mattr=+reference-types -filetype=obj < %s | obj2yaml | FileCheck -check-prefix=BIN %s

test0:
    .functype   test0 () -> ()
    i32.const 42
    f64.const 2.5
    i32.const   0
    call_indirect empty_fref_table, (i32, f64) -> ()
    end_function

.tabletype empty_fref_table, funcref
empty_fref_table:


# CHECK:           .text
# CHECK-LABEL: test0:
# CHECK-NEXT:      .functype   test0 () -> ()
# CHECK-NEXT:      i32.const   42
# CHECK-NEXT:      f64.const   0x1.4p1
# CHECK-NEXT:      i32.const   0
# CHECK-NEXT:      call_indirect empty_fref_table, (i32, f64) -> ()
# CHECK-NEXT:      end_function

# CHECK:           .tabletype empty_fref_table, funcref
# CHECK: empty_fref_table:

# BIN: --- !WASM
# BIN-NEXT: FileHeader:
# BIN-NEXT:   Version:         0x1
# BIN-NEXT: Sections:
# BIN-NEXT:   - Type:            TYPE
# BIN-NEXT:     Signatures:
# BIN-NEXT:       - Index:           0
# BIN-NEXT:         ParamTypes:      []
# BIN-NEXT:         ReturnTypes:     []
# BIN-NEXT:       - Index:           1
# BIN-NEXT:         ParamTypes:
# BIN-NEXT:           - I32
# BIN-NEXT:           - F64
# BIN-NEXT:         ReturnTypes:     []
# BIN-NEXT:   - Type:            IMPORT
# BIN-NEXT:     Imports:
# BIN-NEXT:       - Module:          env
# BIN-NEXT:         Field:           __linear_memory
# BIN-NEXT:         Kind:            MEMORY
# BIN-NEXT:         Memory:
# BIN-NEXT:           Initial:         0x0
# BIN-NEXT:   - Type:            FUNCTION
# BIN-NEXT:     FunctionTypes:   [ 0 ]
# BIN-NEXT:   - Type:            TABLE
# BIN-NEXT:     Tables:
# BIN-NEXT:       - Index:           0
# BIN-NEXT:         ElemType:        FUNCREF
# BIN-NEXT:         Limits:
# BIN-NEXT:           Initial:         0x0
# BIN-NEXT:   - Type:            CODE
# BIN-NEXT:     Relocations:
# BIN-NEXT:       - Type:            R_WASM_TYPE_INDEX_LEB
# BIN-NEXT:         Index:           1
# BIN-NEXT:         Offset:          0x11
# BIN-NEXT:       - Type:            R_WASM_TABLE_NUMBER_LEB
# BIN-NEXT:         Index:           1
# BIN-NEXT:         Offset:          0x16
# BIN-NEXT:     Functions:
# BIN-NEXT:       - Index:           0
# BIN-NEXT:         Locals:          []
# BIN-NEXT:         Body:            412A440000000000000440410011818080800080808080000B
# BIN-NEXT:   - Type:            CUSTOM
# BIN-NEXT:     Name:            linking
# BIN-NEXT:     Version:         2
# BIN-NEXT:     SymbolTable:
# BIN-NEXT:       - Index:           0
# BIN-NEXT:         Kind:            FUNCTION
# BIN-NEXT:         Name:            test0
# BIN-NEXT:         Flags:           [ BINDING_LOCAL ]
# BIN-NEXT:         Function:        0
# BIN-NEXT:       - Index:           1
# BIN-NEXT:         Kind:            TABLE
# BIN-NEXT:         Name:            empty_fref_table
# BIN-NEXT:         Flags:           [ BINDING_LOCAL ]
# BIN-NEXT:         Table:           0
# BIN-NEXT: ...
