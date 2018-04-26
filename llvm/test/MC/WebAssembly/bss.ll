; RUN: llc -filetype=obj %s -o - | obj2yaml | FileCheck %s

target triple = "wasm32-unknown-unknown-wasm"

@g0 = global i8* null, align 4
@g1 = global i32 0, align 4

%union.u1 = type {}
@foo = global %union.u1 zeroinitializer, align 1
@bar = global %union.u1 zeroinitializer, align 1

; CHECK:        - Type:            DATA
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
; CHECK-NEXT:         Content:         ''
; CHECK-NEXT:       - SectionOffset:   29
; CHECK-NEXT:         MemoryIndex:     0
; CHECK-NEXT:         Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           8
; CHECK-NEXT:         Content:         ''

; CHECK:          SymbolTable:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Kind:            DATA
; CHECK-NEXT:         Name:            g0
; CHECK-NEXT:         Flags:           [  ]
; CHECK-NEXT:         Segment:         0
; CHECK-NEXT:         Size:            4
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Kind:            DATA
; CHECK-NEXT:         Name:            g1
; CHECK-NEXT:         Flags:           [  ]
; CHECK-NEXT:         Segment:         1
; CHECK-NEXT:         Size:            4
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Kind:            DATA
; CHECK-NEXT:         Name:            foo
; CHECK-NEXT:         Flags:           [  ]
; CHECK-NEXT:         Segment:         2
; CHECK-NEXT:         Size:            0
; CHECK-NEXT:       - Index:           3
; CHECK-NEXT:         Kind:            DATA
; CHECK-NEXT:         Name:            bar
; CHECK-NEXT:         Flags:           [  ]
; CHECK-NEXT:         Segment:         3
; CHECK-NEXT:         Size:            0
; CHECK-NEXT:     SegmentInfo:    
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            .bss.g0
; CHECK-NEXT:         Alignment:       4
; CHECK-NEXT:         Flags:           [ ]
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Name:            .bss.g1
; CHECK-NEXT:         Alignment:       4
; CHECK-NEXT:         Flags:           [ ]
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Name:            .bss.foo
; CHECK-NEXT:         Alignment:       1
; CHECK-NEXT:         Flags:           [ ]
; CHECK-NEXT:       - Index:           3
; CHECK-NEXT:         Name:            .bss.bar
; CHECK-NEXT:         Alignment:       1
; CHECK-NEXT:         Flags:           [ ]
; CHECK-NEXT: ...
