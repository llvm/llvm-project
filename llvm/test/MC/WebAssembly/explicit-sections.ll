; RUN: llc -mcpu=mvp -filetype=obj %s -o - | obj2yaml | FileCheck %s

target triple = "wasm32-unknown-unknown"

%struct.bd = type { i32, i8 }

@global0 = global i32 8, align 8
@global1 = global %struct.bd  { i32 1, i8 3 }, align 8, section ".sec1"
@global2 = global i64 7, align 8, section ".sec1"
@global3 = global i32 8, align 8, section ".sec2"


; CHECK:        - Type:            DATA{{$}}
; CHECK-NEXT:     Segments:
; CHECK-NEXT:       - SectionOffset:   6
; CHECK-NEXT:         InitFlags:       0
; CHECK-NEXT:         Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           0
; CHECK-NEXT:         Content:         '08000000'
; CHECK-NEXT:       - SectionOffset:   15
; CHECK-NEXT:         InitFlags:       0
; CHECK-NEXT:         Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           8
; CHECK-NEXT:         Content:         '01000000030000000700000000000000'
; CHECK-NEXT:       - SectionOffset:   36
; CHECK-NEXT:         InitFlags:       0
; CHECK-NEXT:         Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           24
; CHECK-NEXT:         Content:         '08000000'

; CHECK:          SymbolTable:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Kind:            DATA
; CHECK-NEXT:         Name:            global0
; CHECK-NEXT:         Flags:           [  ]
; CHECK-NEXT:         Segment:         0
; CHECK-NEXT:         Size:            4
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Kind:            DATA
; CHECK-NEXT:         Name:            global1
; CHECK-NEXT:         Flags:           [  ]
; CHECK-NEXT:         Segment:         1
; CHECK-NEXT:         Size:            8
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Kind:            DATA
; CHECK-NEXT:         Name:            global2
; CHECK-NEXT:         Flags:           [  ]
; CHECK-NEXT:         Segment:         1
; CHECK-NEXT:         Offset:          8
; CHECK-NEXT:         Size:            8
; CHECK-NEXT:       - Index:           3
; CHECK-NEXT:         Kind:            DATA
; CHECK-NEXT:         Name:            global3
; CHECK-NEXT:         Flags:           [  ]
; CHECK-NEXT:         Segment:         2
; CHECK-NEXT:         Size:            4
; CHECK-NEXT:     SegmentInfo:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            .data.global0
; CHECK-NEXT:         Alignment:       3
; CHECK-NEXT:         Flags:           [ ]
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Name:            .sec1
; CHECK-NEXT:         Alignment:       3
; CHECK-NEXT:         Flags:           [ ]
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Name:            .sec2
; CHECK-NEXT:         Alignment:       3
; CHECK-NEXT:         Flags:           [ ]
; CHECK-NEXT: ...
