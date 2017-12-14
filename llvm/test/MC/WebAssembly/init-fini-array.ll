; RUN: llc -mtriple wasm32-unknown-unknown-wasm -filetype=obj %s -o - | obj2yaml | FileCheck %s

@global1 = global i32 1025, align 8

declare void @func1()

declare void @func2()

@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @func1, i8* null }] 

@llvm.global_dtors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @func2, i8* null }]
  

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
; CHECK-NEXT:         Field:           func1
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         SigIndex:        0
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           func2
; CHECK-NEXT:         Kind:            FUNCTION
; CHECK-NEXT:         SigIndex:        0
; CHECK-NEXT:   - Type:            GLOBAL
; CHECK-NEXT:     Globals:         
; CHECK-NEXT:       - Type:            I32
; CHECK-NEXT:         Mutable:         false
; CHECK-NEXT:         InitExpr:        
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           0
; CHECK-NEXT:   - Type:            EXPORT
; CHECK-NEXT:     Exports:         
; CHECK-NEXT:       - Name:            global1
; CHECK-NEXT:         Kind:            GLOBAL
; CHECK-NEXT:         Index:           0
; CHECK-NEXT:   - Type:            ELEM
; CHECK-NEXT:     Segments:        
; CHECK-NEXT:       - Offset:          
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           0
; CHECK-NEXT:         Functions:       [ 0, 1 ]
; CHECK-NEXT:   - Type:            DATA
; CHECK-NEXT:     Relocations:     
; CHECK-NEXT:       - Type:            R_WEBASSEMBLY_TABLE_INDEX_I32
; CHECK-NEXT:         Index:           0
; CHECK-NEXT:         Offset:          0x0000000F
; CHECK-NEXT:       - Type:            R_WEBASSEMBLY_TABLE_INDEX_I32
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x00000018
; CHECK-NEXT:     Segments:        
; CHECK-NEXT:       - SectionOffset:   6
; CHECK-NEXT:         MemoryIndex:     0
; CHECK-NEXT:         Offset:          
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           0
; CHECK-NEXT:         Content:         '01040000'
; CHECK-NEXT:       - SectionOffset:   15
; CHECK-NEXT:         MemoryIndex:     0
; CHECK-NEXT:         Offset:          
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           4
; CHECK-NEXT:         Content:         '00000000'
; CHECK-NEXT:       - SectionOffset:   24
; CHECK-NEXT:         MemoryIndex:     0
; CHECK-NEXT:         Offset:          
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           8
; CHECK-NEXT:         Content:         '01000000'
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            name
; CHECK-NEXT:     FunctionNames:   
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            func1
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Name:            func2
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     DataSize:        12
; CHECK-NEXT:     SegmentInfo:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            .data.global1
; CHECK-NEXT:         Alignment:       8
; CHECK-NEXT:         Flags:           [ ]
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Name:            .init_array
; CHECK-NEXT:         Alignment:       4
; CHECK-NEXT:         Flags:           [ ]
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Name:            .fini_array
; CHECK-NEXT:         Alignment:       4
; CHECK-NEXT:         Flags:           [ ]
; CHECK-NEXT: ...

