; RUN; llc -mcpu=mvp -filetype=obj %s -o - | obj2yaml | FileCheck %s
; RUN; llc -mcpu=mvp -filetype=asm %s -asm-verbose=false -o -  | FileCheck --check-prefix=ASM %s
; RUN: llc -mcpu=mvp -filetype=asm %s -o - | llvm-mc -triple=wasm32 -filetype=obj  -o - | obj2yaml | FileCheck %s

target triple = "wasm32-unknown-unknown"
;target triple = "x86_64-redhat-linux-gnu"
@b = common dso_local global [10 x i32] zeroinitializer, align 4
@c = common dso_local global [20 x i32] zeroinitializer, align 32

; CHECK-ASM: .file	"common.ll"
; CHECK-ASM: .type	b,@object
; CHECK-ASM: .comm	b,40,2
; CHECK-ASM: .type	c,@object
; CHECK-ASM: .comm	c,80,5


; CHECK:      --- !WASM
; CHECK-NEXT: FileHeader:
; CHECK-NEXT:   Version:         0x1
; CHECK-NEXT: Sections:
; CHECK-NEXT:   - Type:            IMPORT
; CHECK-NEXT:     Imports:
; CHECK-NEXT:       - Module:          env
; CHECK-NEXT:         Field:           __linear_memory
; CHECK-NEXT:         Kind:            MEMORY
; CHECK-NEXT:         Memory:
; CHECK-NEXT:           Minimum:         0x1
; CHECK-NEXT:   - Type:            DATACOUNT
; CHECK-NEXT:     Count:           2
; CHECK-NEXT:   - Type:            DATA
; CHECK-NEXT:     Segments:
; CHECK-NEXT:       - SectionOffset:   6
; CHECK-NEXT:         InitFlags:       0
; CHECK-NEXT:         Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           0
; CHECK-NEXT:         Content:         '00000000000000000000000000000000000000000000000000000000000000000000000000000000'
; CHECK-NEXT:       - SectionOffset:   52
; CHECK-NEXT:         InitFlags:       0
; CHECK-NEXT:         Offset:
; CHECK-NEXT:           Opcode:          I32_CONST
; CHECK-NEXT:           Value:           64
; CHECK-NEXT:         Content:         '0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000'
; CHECK-NEXT:   - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     Version:         2
; CHECK-NEXT:     SymbolTable:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Kind:            DATA
; CHECK-NEXT:         Name:            b
; CHECK-NEXT:         Flags:           [ BINDING_WEAK ]
; CHECK-NEXT:         Segment:         0
; CHECK-NEXT:         Size:            40
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Kind:            DATA
; CHECK-NEXT:         Name:            c
; CHECK-NEXT:         Flags:           [ BINDING_WEAK ]
; CHECK-NEXT:         Segment:         1
; CHECK-NEXT:         Size:            80
; CHECK-NEXT:     SegmentInfo:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            .bss.common.b
; CHECK-NEXT:         Alignment:       2
; CHECK-NEXT:         Flags:           [  ]
; CHECK-NEXT:       - Index:           1
; CHECK-NEXT:         Name:            .bss.common.c
; CHECK-NEXT:         Alignment:       5
; CHECK-NEXT:         Flags:           [  ]
; CHECK-NEXT: ...
