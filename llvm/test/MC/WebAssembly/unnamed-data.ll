; RUN: llc -mtriple wasm32-unknown-unknown-wasm -filetype=obj %s -o - | obj2yaml | FileCheck %s

@.str1 = private unnamed_addr constant [6 x i8] c"hello\00", align 1
@.str2 = private unnamed_addr constant [6 x i8] c"world\00", align 1

@a = global i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str1, i32 0, i32 0), align 8
@b = global i8* getelementptr inbounds ([6 x i8], [6 x i8]* @.str2, i32 0, i32 0), align 8


; CHECK:        - Type:            GLOBAL
; CHECK-NEXT:     Globals:         
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Type:            I32
; CHECK-NEXT:         Mutable:         false
; CHECK-NEXT:         InitExpr:        
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           0
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Type:            I32
; CHECK-NEXT:         Mutable:         false
; CHECK-NEXT:         InitExpr:        
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           6
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Type:            I32
; CHECK-NEXT:         Mutable:         false
; CHECK-NEXT:         InitExpr:        
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           16
; CHECK-NEXT:       - Index:           3
; CHECK-NEXT:         Type:            I32
; CHECK-NEXT:         Mutable:         false
; CHECK-NEXT:         InitExpr:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           24
; CHECK-NEXT:   - Type:            EXPORT
; CHECK-NEXT:     Exports:
; CHECK-NEXT:       - Name:            .L.str1
; CHECK-NEXT:         Kind:            GLOBAL
; CHECK-NEXT:         Index:           0
; CHECK-NEXT:       - Name:            .L.str2
; CHECK-NEXT:         Kind:            GLOBAL
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:       - Name:            a
; CHECK-NEXT:         Kind:            GLOBAL
; CHECK-NEXT:         Index:           2
; CHECK-NEXT:       - Name:            b
; CHECK-NEXT:         Kind:            GLOBAL
; CHECK-NEXT:         Index:           3
; CHECK-NEXT:   - Type:            DATA
; CHECK-NEXT:     Relocations:     
; CHECK-NEXT:       - Type:            R_WEBASSEMBLY_MEMORY_ADDR_I32
; CHECK-NEXT:         Index:           0
; CHECK-NEXT:         Offset:          0x0000001C
; CHECK-NEXT:       - Type:            R_WEBASSEMBLY_MEMORY_ADDR_I32
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x00000025
; CHECK-NEXT:     Segments:        
; CHECK-NEXT:       - SectionOffset:   6
; CHECK-NEXT:         MemoryIndex:     0
; CHECK-NEXT:         Offset:          
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           0
; CHECK-NEXT:         Content:         68656C6C6F00
; CHECK-NEXT:       - SectionOffset:   17
; CHECK-NEXT:         MemoryIndex:     0
; CHECK-NEXT:         Offset:          
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           6
; CHECK-NEXT:         Content:         776F726C6400
; CHECK-NEXT:       - SectionOffset:   28
; CHECK-NEXT:         MemoryIndex:     0
; CHECK-NEXT:         Offset:          
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           16
; CHECK-NEXT:         Content:         '00000000'
; CHECK-NEXT:       - SectionOffset:   37
; CHECK-NEXT:         MemoryIndex:     0
; CHECK-NEXT:         Offset:          
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           24
; CHECK-NEXT:         Content:         '06000000'
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     DataSize:        28
; CHECK-NEXT:     SymbolInfo:      
; CHECK-NEXT:       - Name:            .L.str1
; CHECK-NEXT:         Flags:           [ BINDING_LOCAL ]
; CHECK-NEXT:       - Name:            .L.str2
; CHECK-NEXT:         Flags:           [ BINDING_LOCAL ]
; CHECK-NEXT:     SegmentInfo:    
; CHECK-NEXT:       - Index:       0
; CHECK-NEXT:         Name:        .rodata..L.str1
; CHECK-NEXT:         Alignment:   1
; CHECK-NEXT:         Flags:       [ ]
; CHECK-NEXT:       - Index:       1
; CHECK-NEXT:         Name:        .rodata..L.str2
; CHECK-NEXT:         Alignment:   1
; CHECK-NEXT:         Flags:       [ ]
; CHECK-NEXT:       - Index:       2
; CHECK-NEXT:         Name:        .data.a
; CHECK-NEXT:         Alignment:   8
; CHECK-NEXT:         Flags:       [ ]
; CHECK-NEXT:       - Index:       3
; CHECK-NEXT:         Name:        .data.b
; CHECK-NEXT:         Alignment:   8
; CHECK-NEXT:         Flags:       [ ]
; CHECK_NEXT:   ...
