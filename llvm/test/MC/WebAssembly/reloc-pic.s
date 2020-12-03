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

load_hidden_data:
    .functype   load_hidden_data () -> (i32)
    global.get  __memory_base
    i32.const   .L.hidden_data@MBREL
    i32.add
    end_function

load_hidden_func:
    .functype   load_hidden_func () -> (i32)
    global.get  __table_base
    i32.const   hidden_func@TBREL
    i32.add
    end_function

hidden_func:
    .functype   hidden_func () -> (i32)
    i32.const 0
    end_function

.section .rodata.hidden_data,"",@
.L.hidden_data:
    .int8 100
    .size .L.hidden_data, 1

#.hidden hidden_func
#.hidden hidden_data
.size default_data, 4
.functype default_func () -> (i32)

# CHECK:      --- !WASM
# CHECK-NEXT: FileHeader:
# CHECK-NEXT:   Version:         0x1
# CHECK-NEXT: Sections:
# CHECK-NEXT:   - Type:            TYPE
# CHECK-NEXT:     Signatures:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         ParamTypes:      []
# CHECK-NEXT:         ReturnTypes:
# CHECK-NEXT:           - I32
# CHECK-NEXT:   - Type:            IMPORT
# CHECK-NEXT:     Imports:
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           __linear_memory
# CHECK-NEXT:         Kind:            MEMORY
# CHECK-NEXT:         Memory:
# CHECK-NEXT:           Initial:         0x1
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           default_func
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         SigIndex:        0
# CHECK-NEXT:       - Module:          env
# CHECK-NEXT:         Field:           __indirect_function_table
# CHECK-NEXT:         Kind:            TABLE
# CHECK-NEXT:         Table:
# CHECK-NEXT:           Index:           0
# CHECK-NEXT:           ElemType:        FUNCREF
# CHECK-NEXT:           Limits:
# CHECK-NEXT:             Initial:         0x1
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
# CHECK-NEXT:     FunctionTypes:   [ 0, 0, 0, 0, 0 ]
# CHECK-NEXT:   - Type:            ELEM
# CHECK-NEXT:     Segments:
# CHECK-NEXT:        Offset:
# CHECK-NEXT:          Opcode:          I32_CONST
# CHECK-NEXT:          Value:           1
# CHECK-NEXT:        Functions:       [ 5 ]
# CHECK-NEXT:   - Type:            DATACOUNT
# CHECK-NEXT:     Count:           1
# CHECK-NEXT:   - Type:            CODE
# CHECK-NEXT:     Relocations:
# CHECK-NEXT:       - Type:            R_WASM_GLOBAL_INDEX_LEB
# CHECK-NEXT:         Index:           1
# CHECK-NEXT:         Offset:          0x4
# CHECK-NEXT:       - Type:            R_WASM_GLOBAL_INDEX_LEB
# CHECK-NEXT:         Index:           3
# CHECK-NEXT:         Offset:          0x10
# CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
# CHECK-NEXT:         Index:           5
# CHECK-NEXT:         Offset:          0x1C
# CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_REL_SLEB
# CHECK-NEXT:         Index:           6
# CHECK-NEXT:         Offset:          0x22
# CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_LEB
# CHECK-NEXT:         Index:           8
# CHECK-NEXT:         Offset:          0x2C
# CHECK-NEXT:       - Type:            R_WASM_TABLE_INDEX_REL_SLEB
# CHECK-NEXT:         Index:           9
# CHECK-NEXT:         Offset:          0x32
# CHECK-NEXT:     Functions:
# CHECK-NEXT:       - Index:           1
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            2380808080002802000B
# CHECK-NEXT:       - Index:           2
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            2381808080002802000B
# CHECK-NEXT:       - Index:           3
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            2380808080004180808080006A0B
# CHECK-NEXT:       - Index:           4
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            2380808080004180808080006A0B
# CHECK-NEXT:       - Index:           5
# CHECK-NEXT:         Locals:          []
# CHECK-NEXT:         Body:            41000B
# CHECK-NEXT:   - Type:            DATA
# CHECK-NEXT:     Segments:
# CHECK-NEXT:       - SectionOffset:   6
# CHECK-NEXT:         InitFlags:       0
# CHECK-NEXT:         Offset:
# CHECK-NEXT:           Opcode:          I32_CONST
# CHECK-NEXT:           Value:           0
# CHECK-NEXT:         Content:         '64'
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
# CHECK-NEXT:       - Index:           4
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Name:            load_hidden_data
# CHECK-NEXT:         Flags:           [ BINDING_LOCAL ]
# CHECK-NEXT:         Function:        3
# CHECK-NEXT:       - Index:           5
# CHECK-NEXT:         Kind:            DATA
# CHECK-NEXT:         Name:            __memory_base
# CHECK-NEXT:         Flags:           [ UNDEFINED ]
# CHECK-NEXT:       - Index:           6
# CHECK-NEXT:         Kind:            DATA
# CHECK-NEXT:         Name:            .L.hidden_data
# CHECK-NEXT:         Flags:           [ BINDING_LOCAL ]
# CHECK-NEXT:         Segment:         0
# CHECK-NEXT:         Size:            1
# CHECK-NEXT:       - Index:           7
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Name:            load_hidden_func
# CHECK-NEXT:         Flags:           [ BINDING_LOCAL ]
# CHECK-NEXT:         Function:        4
# CHECK-NEXT:       - Index:           8
# CHECK-NEXT:         Kind:            DATA
# CHECK-NEXT:         Name:            __table_base
# CHECK-NEXT:         Flags:           [ UNDEFINED ]
# CHECK-NEXT:       - Index:           9
# CHECK-NEXT:         Kind:            FUNCTION
# CHECK-NEXT:         Name:            hidden_func
# CHECK-NEXT:         Flags:           [ BINDING_LOCAL ]
# CHECK-NEXT:         Function:        5
# CHECK-NEXT:     SegmentInfo:
# CHECK-NEXT:       - Index:           0
# CHECK-NEXT:         Name:            .rodata.hidden_data
# CHECK-NEXT:         Alignment:       0
# CHECK-NEXT:         Flags:           [  ]
# CHECK-NEXT: ...
