; RUN: llc -mtriple wasm32-unknown-unknown-wasm -filetype=obj %s -o - | obj2yaml | FileCheck %s

@global1 = global i32 1025, align 8

declare void @func0()
declare void @func1()
declare void @func2()
declare void @func3()

@llvm.global_ctors = appending global [2 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @func0, i8* null }, { i32, void ()*, i8* } { i32 42, void ()* @func1, i8* null }] 

@llvm.global_dtors = appending global [2 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @func2, i8* null }, { i32, void ()*, i8* } { i32 42, void ()* @func3, i8* null }] 
  

; CHECK:        - Type:            IMPORT
; CHECK-NEXT:     Imports:         
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           __linear_memory
; CHECK-NEXT:         Kind:            MEMORY
; CHECK-NEXT:         Memory:          
; CHECK-NEXT:           Initial:         0x00000001
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           __indirect_function_table
; CHECK-NEXT:         Kind:            TABLE
; CHECK-NEXT:         Table:           
; CHECK-NEXT:           ElemType:        ANYFUNC
; CHECK-NEXT:           Limits:          
; CHECK-NEXT:             Initial:         0x00000002
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           func3
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         SigIndex:        1
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           __dso_handle
; CHECK-NEXT:         Kind:            GLOBAL
; CHECK-NEXT:         GlobalType:      I32
; CHECK-NEXT:         GlobalMutable:   false
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
; CHECK-NEXT:   - Type:            FUNCTION
; CHECK-NEXT:     FunctionTypes:   [ 0, 1, 0, 1 ]
; CHECK-NEXT:   - Type:            GLOBAL
; CHECK-NEXT:     Globals:         
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Type:            I32
; CHECK-NEXT:         Mutable:         false
; CHECK-NEXT:         InitExpr:        
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           0
; CHECK-NEXT:   - Type:            EXPORT
; CHECK-NEXT:     Exports:         
; CHECK-NEXT:       - Name:            .Lcall_dtors.42
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           5
; CHECK-NEXT:       - Name:            .Lregister_call_dtors.42
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           6
; CHECK-NEXT:       - Name:            .Lcall_dtors
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           7
; CHECK-NEXT:       - Name:            .Lregister_call_dtors
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         Index:           8
; CHECK-NEXT:       - Name:            global1
; CHECK-NEXT:         Kind:            GLOBAL
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:   - Type:            ELEM
; CHECK-NEXT:     Segments:        
; CHECK-NEXT:       - Offset:          
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           0
; CHECK-NEXT:         Functions:       [ 5, 7 ]
; CHECK-NEXT:   - Type:            CODE
; CHECK-NEXT:     Relocations:     
; CHECK-NEXT:       - Type:            R_WEBASSEMBLY_FUNCTION_INDEX_LEB
; CHECK-NEXT:         Index:           0
; CHECK-NEXT:         Offset:          0x00000004
; CHECK-NEXT:       - Type:            R_WEBASSEMBLY_TABLE_INDEX_SLEB
; CHECK-NEXT:         Index:           0
; CHECK-NEXT:         Offset:          0x0000000F
; CHECK-NEXT:       - Type:            R_WEBASSEMBLY_MEMORY_ADDR_SLEB
; CHECK-NEXT:         Index:           0
; CHECK-NEXT:         Offset:          0x00000017
; CHECK-NEXT:       - Type:            R_WEBASSEMBLY_FUNCTION_INDEX_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x0000001D
; CHECK-NEXT:       - Type:            R_WEBASSEMBLY_FUNCTION_INDEX_LEB
; CHECK-NEXT:         Index:           2
; CHECK-NEXT:         Offset:          0x0000002C
; CHECK-NEXT:       - Type:            R_WEBASSEMBLY_TABLE_INDEX_SLEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x00000037
; CHECK-NEXT:       - Type:            R_WEBASSEMBLY_MEMORY_ADDR_SLEB
; CHECK-NEXT:         Index:           0
; CHECK-NEXT:         Offset:          0x0000003F
; CHECK-NEXT:       - Type:            R_WEBASSEMBLY_FUNCTION_INDEX_LEB
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x00000045
; CHECK-NEXT:     Functions:       
; CHECK-NEXT:       - Index:           5
; CHECK-NEXT:         Locals:          
; CHECK-NEXT:         Body:            1080808080000B
; CHECK-NEXT:       - Index:           6
; CHECK-NEXT:         Locals:          
; CHECK-NEXT:         Body:            0240418080808000410041FFFFFFFF7F1081808080000D000F0B00000B
; CHECK-NEXT:       - Index:           7
; CHECK-NEXT:         Locals:          
; CHECK-NEXT:         Body:            1082808080000B
; CHECK-NEXT:       - Index:           8
; CHECK-NEXT:         Locals:          
; CHECK-NEXT:         Body:            0240418180808000410041FFFFFFFF7F1081808080000D000F0B00000B
; CHECK-NEXT:   - Type:            DATA
; CHECK-NEXT:     Segments:        
; CHECK-NEXT:       - SectionOffset:   6
; CHECK-NEXT:         MemoryIndex:     0
; CHECK-NEXT:         Offset:          
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           0
; CHECK-NEXT:         Content:         '01040000'
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            name
; CHECK-NEXT:     FunctionNames:   
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            func3
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Name:            __cxa_atexit
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Name:            func2
; CHECK-NEXT:       - Index:           3
; CHECK-NEXT:         Name:            func1
; CHECK-NEXT:       - Index:           4
; CHECK-NEXT:         Name:            func0
; CHECK-NEXT:       - Index:           5
; CHECK-NEXT:         Name:            .Lcall_dtors.42
; CHECK-NEXT:       - Index:           6
; CHECK-NEXT:         Name:            .Lregister_call_dtors.42
; CHECK-NEXT:       - Index:           7
; CHECK-NEXT:         Name:            .Lcall_dtors
; CHECK-NEXT:       - Index:           8
; CHECK-NEXT:         Name:            .Lregister_call_dtors
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     DataSize:        4
; CHECK-NEXT:     SymbolInfo:      
; CHECK-NEXT:       - Name:            __dso_handle
; CHECK-NEXT:         Flags:           [ BINDING_WEAK, VISIBILITY_HIDDEN ]
; CHECK-NEXT:       - Name:            .Lcall_dtors.42
; CHECK-NEXT:         Flags:           [ BINDING_LOCAL ]
; CHECK-NEXT:       - Name:            .Lregister_call_dtors.42
; CHECK-NEXT:         Flags:           [ BINDING_LOCAL ]
; CHECK-NEXT:       - Name:            .Lcall_dtors
; CHECK-NEXT:         Flags:           [ BINDING_LOCAL ]
; CHECK-NEXT:       - Name:            .Lregister_call_dtors
; CHECK-NEXT:         Flags:           [ BINDING_LOCAL ]
; CHECK-NEXT:     SegmentInfo:     
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            .data.global1
; CHECK-NEXT:         Alignment:       8
; CHECK-NEXT:         Flags:           [ ]
; CHECK-NEXT:     InitFunctions:     
; CHECK-NEXT:       - Priority: 42
; CHECK-NEXT:         FunctionIndex: 3
; CHECK-NEXT:       - Priority: 42
; CHECK-NEXT:         FunctionIndex: 6
; CHECK-NEXT:       - Priority: 65535
; CHECK-NEXT:         FunctionIndex: 4
; CHECK-NEXT:       - Priority: 65535
; CHECK-NEXT:         FunctionIndex: 8
; CHECK-NEXT: ...
