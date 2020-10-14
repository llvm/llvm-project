# RUN: llvm-mc -triple=wasm32-unknown-unknown -mattr=+unimplemented-simd128,+nontrapping-fptoint,+exception-handling < %s | FileCheck %s
# Check that it converts to .o without errors:
# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj -mattr=+unimplemented-simd128,+nontrapping-fptoint,+exception-handling < %s | obj2yaml | FileCheck -check-prefix=BIN %s

# Minimal test for type indices in call_indirect.

test0:
    .functype   test0 (i32) -> (i32)
    call_indirect (f64) -> (f64)
    end_function

# CHECK:	.text
# CHECK-LABEL: test0:
# CHECK-NEXT:	.functype	test0 (i32) -> (i32)
# CHECK-NEXT:	call_indirect	(f64) -> (f64)
# CHECK-NEXT:	end_function

# BIN:      --- !WASM
# BIN-NEXT: FileHeader:
# BIN-NEXT:   Version:         0x00000001
# BIN-NEXT: Sections:
# BIN-NEXT:   - Type:            TYPE
# BIN-NEXT:     Signatures:
# BIN-NEXT:       - Index:           0
# BIN-NEXT:         ParamTypes:
# BIN-NEXT:           - I32
# BIN-NEXT:         ReturnTypes:
# BIN-NEXT:           - I32
# BIN-NEXT:       - Index:           1
# BIN-NEXT:         ParamTypes:
# BIN-NEXT:           - F64
# BIN-NEXT:         ReturnTypes:
# BIN-NEXT:           - F64
# BIN-NEXT:   - Type:            IMPORT
# BIN-NEXT:     Imports:
# BIN-NEXT:       - Module:          env
# BIN-NEXT:         Field:           __linear_memory
# BIN-NEXT:         Kind:            MEMORY
# BIN-NEXT:         Memory:
# BIN-NEXT:           Initial:         0x00000000
# BIN-NEXT:       - Module:          env
# BIN-NEXT:         Field:           __indirect_function_table
# BIN-NEXT:         Kind:            TABLE
# BIN-NEXT:         Table:
# BIN-NEXT:           Index:           0
# BIN-NEXT:           ElemType:        FUNCREF
# BIN-NEXT:           Limits:
# BIN-NEXT:             Initial:         0x00000000
# BIN-NEXT:   - Type:            FUNCTION
# BIN-NEXT:     FunctionTypes:   [ 0 ]
# BIN-NEXT:   - Type:            CODE
# BIN-NEXT:     Relocations:
# BIN-NEXT:       - Type:            R_WASM_TYPE_INDEX_LEB
# BIN-NEXT:         Index:           1
# BIN-NEXT:         Offset:          0x00000004
# BIN-NEXT:     Functions:
# BIN-NEXT:       - Index:           0
# BIN-NEXT:         Locals:          []
# BIN-NEXT:         Body:            118180808000000B
# BIN-NEXT:   - Type:            CUSTOM
# BIN-NEXT:     Name:            linking
# BIN-NEXT:     Version:         2
# BIN-NEXT:     SymbolTable:
# BIN-NEXT:       - Index:           0
# BIN-NEXT:         Kind:            FUNCTION
# BIN-NEXT:         Name:            test0
# BIN-NEXT:         Flags:           [ BINDING_LOCAL ]
# BIN-NEXT:         Function:        0
# BIN-NEXT: ...
