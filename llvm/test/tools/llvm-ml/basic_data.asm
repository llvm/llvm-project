; RUN: llvm-ml -filetype=s %s /Fo - | FileCheck %s

.data
BYTE 2, 4, 6, 8
; CHECK: .data
; CHECK-NEXT: .byte	2
; CHECK-NEXT: .byte	4
; CHECK-NEXT: .byte	6
; CHECK-NEXT: .byte	8

BYTE 2 dup (1, 2 dup (2)),
     3
; CHECK: .byte	1
; CHECK-NEXT: .byte	2
; CHECK-NEXT: .byte	2
; CHECK-NEXT: .byte	1
; CHECK-NEXT: .byte	2
; CHECK-NEXT: .byte	2
; CHECK-NEXT: .byte	3

REAL4 1, 0
; CHECK: .long 1065353216
; CHECK-NEXT: .long 0

REAL4 2 DUP (2.5, 2 dup (0)),
      4
; CHECK: .long 1075838976
; CHECK-NEXT: .long 0
; CHECK-NEXT: .long 0
; CHECK-NEXT: .long 1075838976
; CHECK-NEXT: .long 0
; CHECK-NEXT: .long 0
; CHECK-NEXT: .long 1082130432

.code
BYTE 5
; CHECK: .text
; CHECK-NEXT: .byte	5
