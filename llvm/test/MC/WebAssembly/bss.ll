; RUN: llc -mtriple wasm32-unknown-unknown-wasm -filetype=obj %s -o - | obj2yaml | FileCheck %s

@g0 = global i8* null, align 4

; CHECK:        - Type:            DATA
; CHECK-NEXT:     Segments:        
; CHECK-NEXT:       - SectionOffset:   6
; CHECK-NEXT:         MemoryIndex:     0
; CHECK-NEXT:         Offset:          
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           0
; CHECK-NEXT:         Content:         '00000000'
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     DataSize:        4
; CHECK-NEXT:     DataAlignment:   4
; CHECK-NEXT:     SegmentNames:    
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            .bss.g0
; CHECK-NEXT: ...
