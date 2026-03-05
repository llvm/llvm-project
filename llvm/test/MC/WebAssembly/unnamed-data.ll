; RUN: llc -mcpu=mvp -filetype=obj %s -o - | obj2yaml | FileCheck %s

target triple = "wasm32-unknown-unknown"

@.str1 = private unnamed_addr constant [6 x i8] c"hello\00", align 1
@.str2 = private unnamed_addr constant [6 x i8] c"world\00", align 1

@a = global ptr @.str1, align 8
@b = global ptr @.str2, align 8


; CHECK:        - Type:            DATA{{$}}
; CHECK-NEXT:     Relocations:
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_I32
; CHECK-NEXT:         Index:           0
; CHECK-NEXT:         Offset:          0x1C
; CHECK-NEXT:       - Type:            R_WASM_MEMORY_ADDR_I32
; CHECK-NEXT:         Index:           1
; CHECK-NEXT:         Offset:          0x25
; CHECK-NEXT:     Segments:
; CHECK-NEXT:       - SectionOffset:   6
; CHECK-NEXT:         InitFlags:       0
; CHECK-NEXT:         Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           0
; CHECK-NEXT:         Content:         68656C6C6F00
; CHECK-NEXT:       - SectionOffset:   17
; CHECK-NEXT:         InitFlags:       0
; CHECK-NEXT:         Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           6
; CHECK-NEXT:         Content:         776F726C6400
; CHECK-NEXT:       - SectionOffset:   28
; CHECK-NEXT:         InitFlags:       0
; CHECK-NEXT:         Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           16
; CHECK-NEXT:         Content:         '00000000'
; CHECK-NEXT:       - SectionOffset:   37
; CHECK-NEXT:         InitFlags:       0
; CHECK-NEXT:         Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           24
; CHECK-NEXT:         Content:         '06000000'
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     Version:         2
; CHECK-NEXT:     SymbolTable:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Kind:            DATA
; CHECK-NEXT:         Name:            .L.str1
; CHECK-NEXT:         Flags:           [ BINDING_LOCAL ]
; CHECK-NEXT:         Segment:         0
; CHECK-NEXT:         Size:            6
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Kind:            DATA
; CHECK-NEXT:         Name:            .L.str2
; CHECK-NEXT:         Flags:           [ BINDING_LOCAL ]
; CHECK-NEXT:         Segment:         1
; CHECK-NEXT:         Size:            6
; CHECK-NEXT:       - Index:           2
; CHECK-NEXT:         Kind:            DATA
; CHECK-NEXT:         Name:            a
; CHECK-NEXT:         Flags:           [  ]
; CHECK-NEXT:         Segment:         2
; CHECK-NEXT:         Size:            4
; CHECK-NEXT:       - Index:           3
; CHECK-NEXT:         Kind:            DATA
; CHECK-NEXT:         Name:            b
; CHECK-NEXT:         Flags:           [  ]
; CHECK-NEXT:         Segment:         3
; CHECK-NEXT:         Size:            4
; CHECK-NEXT:     SegmentInfo:
; CHECK-NEXT:       - Index:       0
; CHECK-NEXT:         Name:        .rodata..L.str1
; CHECK-NEXT:         Alignment:   0
; CHECK-NEXT:         Flags:       [ STRINGS ]
; CHECK-NEXT:       - Index:       1
; CHECK-NEXT:         Name:        .rodata..L.str2
; CHECK-NEXT:         Alignment:   0
; CHECK-NEXT:         Flags:       [ STRINGS ]
; CHECK-NEXT:       - Index:       2
; CHECK-NEXT:         Name:        .data.a
; CHECK-NEXT:         Alignment:   3
; CHECK-NEXT:         Flags:       [ ]
; CHECK-NEXT:       - Index:       3
; CHECK-NEXT:         Name:        .data.b
; CHECK-NEXT:         Alignment:   3
; CHECK-NEXT:         Flags:       [ ]
; CHECK-NEXT:   ...
