; RUN: llc -filetype=obj %s -o - | obj2yaml | FileCheck %s

target triple = "wasm32-unknown-unknown"

@global1 = global i32 1025, align 8

declare void @func0()
declare void @func1()
declare void @func2()
declare void @func3()

@llvm.global_ctors = appending global [2 x { i32, void ()*, i8* }] [
  { i32, void ()*, i8* } { i32 65535, void ()* @func0, i8* null },
  { i32, void ()*, i8* } { i32 42,    void ()* @func1, i8* null }
]

@llvm.global_dtors = appending global [2 x { i32, void ()*, i8* }] [
  { i32, void ()*, i8* } { i32 65535, void ()* @func2, i8* null },
  { i32, void ()*, i8* } { i32 42,    void ()* @func3, i8* null }
]

; CHECK:        - Type:            IMPORT
; CHECK-NEXT:     Imports:
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           __linear_memory
; CHECK-NEXT:         Kind:            MEMORY
; CHECK-NEXT:         Memory:
; CHECK-NEXT:           Initial:         0x1
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           func3
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         SigIndex:        1
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           __cxa_atexit
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         SigIndex:        2
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           func2
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         SigIndex:        1
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           func1
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         SigIndex:        1
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           func0
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         SigIndex:        1
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           __indirect_function_table
; CHECK-NEXT:         Kind:            TABLE
; CHECK-NEXT:         Table:
; CHECK-NEXT:           Index:           0
; CHECK-NEXT:           ElemType:        FUNCREF
; CHECK-NEXT:           Limits:
; CHECK-NEXT:             Initial:         0x2
; CHECK-NEXT:   - Type:            FUNCTION
; CHECK-NEXT:     FunctionTypes:   [ 0, 1, 0, 1 ]
; CHECK-NEXT:   - Type:            ELEM
; CHECK-NEXT:     Segments:
; CHECK-NEXT:       - Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           1
; CHECK-NEXT:         Functions:       [ 5, 7 ]
; CHECK-NEXT:   - Type:            DATACOUNT
; CHECK-NEXT:     Count:           1
; CHECK-NEXT:   - Type:            CODE
; CHECK-NEXT:     Relocations:
; CHECK-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x4
; CHECK-NEXT:       - Type:            R_WASM_TABLE_INDEX_SLEB
; CHECK-NEXT:         Index:           0
; CHECK-NEXT:         Offset:          0xF
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_SLEB
; CHECK-NEXT:         Index:           3
; CHECK-NEXT:         Offset:          0x17
; CHECK-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
; CHECK-NEXT:         Index:           4
; CHECK-NEXT:         Offset:          0x1D
; CHECK-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
; CHECK-NEXT:         Index:           6
; CHECK-NEXT:         Offset:          0x2C
; CHECK-NEXT:       - Type:            R_WASM_TABLE_INDEX_SLEB
; CHECK-NEXT:         Index:           5
; CHECK-NEXT:         Offset:          0x37
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_SLEB
; CHECK-NEXT:         Index:           3
; CHECK-NEXT:         Offset:          0x3F
; CHECK-NEXT:       - Type:            R_WASM_FUNCTION_INDEX_LEB
; CHECK-NEXT:         Index:           4
; CHECK-NEXT:         Offset:          0x45
; CHECK-NEXT:     Functions:
; CHECK-NEXT:       - Index:           5
; CHECK-NEXT:         Locals:
; CHECK-NEXT:         Body:            1080808080000B
; CHECK-NEXT:       - Index:           6
; CHECK-NEXT:         Locals:
; CHECK-NEXT:         Body:            02404181808080004100418080808000108180808000450D0000000B0B
; CHECK-NEXT:       - Index:           7
; CHECK-NEXT:         Locals:
; CHECK-NEXT:         Body:            1082808080000B
; CHECK-NEXT:       - Index:           8
; CHECK-NEXT:         Locals:
; CHECK-NEXT:         Body:            02404182808080004100418080808000108180808000450D0000000B0B
; CHECK-NEXT:   - Type:            DATA
; CHECK-NEXT:     Segments:
; CHECK-NEXT:       - SectionOffset:   6
; CHECK-NEXT:         InitFlags:       0
; CHECK-NEXT:         Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           0
; CHECK-NEXT:         Content:         '01040000'
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     Version:         2
; CHECK-NEXT:     SymbolTable:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Name:            .Lcall_dtors.42
; CHECK-NEXT:         Flags:           [ BINDING_LOCAL ]
; CHECK-NEXT:         Function:        5
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Name:            func3
; CHECK-NEXT:         Flags:           [ UNDEFINED ]
; CHECK-NEXT:         Function:        0
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Name:            .Lregister_call_dtors.42
; CHECK-NEXT:         Flags:           [ BINDING_LOCAL ]
; CHECK-NEXT:         Function:        6
; CHECK-NEXT:       - Index:           3
; CHECK-NEXT:         Kind:            DATA
; CHECK-NEXT:         Name:            __dso_handle
; CHECK-NEXT:         Flags:           [ BINDING_WEAK, VISIBILITY_HIDDEN, UNDEFINED ]
; CHECK-NEXT:       - Index:           4
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Name:            __cxa_atexit
; CHECK-NEXT:         Flags:           [ UNDEFINED ]
; CHECK-NEXT:         Function:        1
; CHECK-NEXT:       - Index:           5
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Name:            .Lcall_dtors
; CHECK-NEXT:         Flags:           [ BINDING_LOCAL ]
; CHECK-NEXT:         Function:        7
; CHECK-NEXT:       - Index:           6
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Name:            func2
; CHECK-NEXT:         Flags:           [ UNDEFINED ]
; CHECK-NEXT:         Function:        2
; CHECK-NEXT:       - Index:           7
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Name:            .Lregister_call_dtors
; CHECK-NEXT:         Flags:           [ BINDING_LOCAL ]
; CHECK-NEXT:         Function:        8
; CHECK-NEXT:       - Index:           8
; CHECK-NEXT:         Kind:            DATA
; CHECK-NEXT:         Name:            global1
; CHECK-NEXT:         Flags:           [  ]
; CHECK-NEXT:         Segment:         0
; CHECK-NEXT:         Size:            4
; CHECK-NEXT:       - Index:           9
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Name:            func1
; CHECK-NEXT:         Flags:           [ UNDEFINED ]
; CHECK-NEXT:         Function:        3
; CHECK-NEXT:       - Index:           10
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Name:            func0
; CHECK-NEXT:         Flags:           [ UNDEFINED ]
; CHECK-NEXT:         Function:        4
; CHECK-NEXT:     SegmentInfo:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            .data.global1
; CHECK-NEXT:         Alignment:       3
; CHECK-NEXT:         Flags:           [ ]
; CHECK-NEXT:     InitFunctions:
; CHECK-NEXT:       - Priority: 42
; CHECK-NEXT:         Symbol: 9
; CHECK-NEXT:       - Priority: 42
; CHECK-NEXT:         Symbol: 2
; CHECK-NEXT:       - Priority: 65535
; CHECK-NEXT:         Symbol: 10
; CHECK-NEXT:       - Priority: 65535
; CHECK-NEXT:         Symbol: 7
; CHECK-NEXT: ...
