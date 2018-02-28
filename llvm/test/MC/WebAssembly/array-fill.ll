; RUN: llc -filetype=obj %s -o - | obj2yaml | FileCheck %s
; PR33624

source_filename = "ws.c"
target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown-wasm"

%struct.bd = type { i8 }

@gBd = hidden global [2 x %struct.bd] [%struct.bd { i8 1 }, %struct.bd { i8 2 }], align 1

; CHECK:        - Type:            DATA
; CHECK:              Content:         '0102'

; CHECK:        - Type:            CUSTOM
; CHECK-NEXT:     Name:            linking
; CHECK-NEXT:     SymbolTable:
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Kind:            DATA
; CHECK-NEXT:         Name:            gBd
; CHECK-NEXT:         Flags:           [ VISIBILITY_HIDDEN ]
; CHECK-NEXT:         Segment:         0
; CHECK-NEXT:         Size:            2
; CHECK-NEXT:     SegmentInfo:    
; CHECK-NEXT:       - Index:           0
; CHECK-NEXT:         Name:            .data
; CHECK-NEXT:         Alignment:       1
; CHECK-NEXT:         Flags:           [ ]
; CHECK-NEXT: ...
