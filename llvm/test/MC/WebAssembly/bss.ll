; RUN: llc -mtriple wasm32-unknown-unknown-wasm -filetype=obj %s -o - | obj2yaml | FileCheck %s

@g0 = global i8* null, align 4
@g1 = global i32 0, align 4

%union.u1 = type {}
@foo = global %union.u1 zeroinitializer, align 1
@bar = global %union.u1 zeroinitializer, align 1

; CHECK:        - Type:            GLOBAL
; CHECK-NEXT:     Globals:         
; CHECK-NEXT:       - Type:            I32
; CHECK-NEXT:         Mutable:         false
; CHECK-NEXT:         InitExpr:        
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           0
; CHECK-NEXT:       - Type:            I32
; CHECK-NEXT:         Mutable:         false
; CHECK-NEXT:         InitExpr:        
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           4
; CHECK-NEXT:       - Type:            I32
; CHECK-NEXT:         Mutable:         false
; CHECK-NEXT:         InitExpr:        
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           8
; CHECK-NEXT:       - Type:            I32
; CHECK-NEXT:         Mutable:         false
; CHECK-NEXT:         InitExpr:        
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           9
; CHECK-NEXT:   - Type:            EXPORT
; CHECK-NEXT:     Exports:         
; CHECK-NEXT:       - Name:            g0
; CHECK-NEXT:         Kind:            GLOBAL
; CHECK-NEXT:         Index:           0
; CHECK-NEXT:       - Name:            g1
; CHECK-NEXT:         Kind:            GLOBAL
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:       - Name:            foo
; CHECK-NEXT:         Kind:            GLOBAL
; CHECK-NEXT:         Index:           2
; CHECK-NEXT:       - Name:            bar
; CHECK-NEXT:         Kind:            GLOBAL
; CHECK-NEXT:         Index:           3
; CHECK-NEXT:   - Type:            DATA
; CHECK-NEXT:     Segments:        
; CHECK-NEXT:       - SectionOffset:   6
; CHECK-NEXT:         MemoryIndex:     0
; CHECK-NEXT:         Offset:          
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           0
; CHECK-NEXT:         Content:         '00000000'
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
; CHECK-NEXT:         Content:         '00'
; CHECK-NEXT:       - SectionOffset:  30 
; CHECK-NEXT:         MemoryIndex:     0
; CHECK-NEXT:         Offset:          
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           9
; CHECK-NEXT:         Content:         '00'
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     DataSize:        10
; CHECK-NEXT:     SegmentInfo:    
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            .bss.g0
; CHECK-NEXT:         Alignment:       4
; CHECK-NEXT:         Flags:           0
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Name:            .bss.g1
; CHECK-NEXT:         Alignment:       4
; CHECK-NEXT:         Flags:           0
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Name:            .bss.foo
; CHECK-NEXT:         Alignment:       1
; CHECK-NEXT:         Flags:           0
; CHECK-NEXT:       - Index:           3
; CHECK-NEXT:         Name:            .bss.bar
; CHECK-NEXT:         Alignment:       1
; CHECK-NEXT:         Flags:           0
; CHECK-NEXT: ...
