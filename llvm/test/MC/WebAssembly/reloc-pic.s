# RUN: llvm-mc -triple=wasm32-unknown-unknown -filetype=obj < %s | obj2yaml | FileCheck %s

# Verify that @GOT relocation entryes result in R_WASM_GLOBAL_INDEX_LEB against
# against the corrsponding function or data symbol and that the corresponding
# data symbols are imported as a wasm globals.

load_default_data:
    .functype   load_default_data () -> (i32)
    global.get  default_data@GOT
    i32.load    0
    end_function

load_default_func:
    .functype   load_default_func () -> (i32)
    global.get  default_func@GOT
    i32.load    0
    end_function

.size default_data, 4
.functype default_func () -> (i32)

# CHECK:      --- !WASM
# CHECK-NEXT: FileHeader:
# CHECK-NEXT:   Version:         0x00000001
# CHECK-NEXT: Sections:
# CHECK-NEXT:   - Type:            TYPE
# CHECK-NEXT:     Signatures:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         ReturnType:      I32
# CHECK-NEXT:         ParamTypes:      []
# CHECK-NEXT:   - Type:            IMPORT
# CHECK-NEXT:     Imports:
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           __linear_memory
# CHECK-NEXT:         Kind:            MEMORY
# CHECK-NEXT:         Memory:
# CHECK-NEXT:           Initial:         0x00000000
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           __indirect_function_table
# CHECK-NEXT:         Kind:            TABLE
# CHECK-NEXT:         Table:
# CHECK-NEXT:           ElemType:        FUNCREF
# CHECK-NEXT:           Limits:
# CHECK-NEXT:             Initial:         0x00000000
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           default_func
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         SigIndex:        0
# CHECK-NEXT:       - Module:          GOT.mem
# CHECK-NEXT:         Field:           default_data
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I32
# CHECK-NEXT:         GlobalMutable:   true
# CHECK-NEXT:       - Module:          GOT.func
# CHECK-NEXT:         Field:           default_func
# CHECK-NEXT:         Kind:            GLOBAL
# CHECK-NEXT:         GlobalType:      I32
# CHECK-NEXT:         GlobalMutable:   true
# CHECK-NEXT:   - Type:            FUNCTION
# CHECK-NEXT:     FunctionTypes:   [ 0, 0 ]
# CHECK-NEXT:   - Type:            CODE
# CHECK-NEXT:     Relocations:
# CHECK-NEXT:       - Type:            R_WASM_GLOBAL_INDEX_LEB
# CHECK-NEXT:         Index:           1
# CHECK-NEXT:         Offset:          0x00000004
# CHECK-NEXT:       - Type:            R_WASM_GLOBAL_INDEX_LEB
# CHECK-NEXT:         Index:           3
# CHECK-NEXT:         Offset:          0x00000010
# CHECK-NEXT:     Functions:
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            2380808080002800000B
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            2381808080002800000B
# CHECK-NEXT:   - Type:            CUSTOM
# CHECK-NEXT:     Name:            linking
# CHECK-NEXT:     Version:         2
# CHECK-NEXT:     SymbolTable:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Name:            load_default_data
# CHECK-NEXT:         Flags:           [ BINDING_LOCAL ]
# CHECK-NEXT:         Function:        1
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Kind:            DATA
# CHECK-NEXT:         Name:            default_data
# CHECK-NEXT:         Flags:           [ UNDEFINED ]
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Name:            load_default_func
# CHECK-NEXT:         Flags:           [ BINDING_LOCAL ]
# CHECK-NEXT:         Function:        2
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Name:            default_func
# CHECK-NEXT:         Flags:           [ UNDEFINED ]
# CHECK-NEXT:         Function:        0
# CHECK-NEXT: ...
