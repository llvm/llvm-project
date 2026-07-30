; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      cmp.b  %d0, %d1
; CHECK-SAME: encoding: [0xb2,0x00]
cmp.b	%d0, %d1
; CHECK:      cmp.b  %d3, %d2
; CHECK-SAME: encoding: [0xb4,0x03]
cmp.b	%d3, %d2
; CHECK:      cmp.w  %d4, %d5
; CHECK-SAME: encoding: [0xba,0x44]
cmp.w	%d4, %d5
; CHECK:      cmp.w  %d2, %d3
; CHECK-SAME: encoding: [0xb6,0x42]
cmp.w	%d2, %d3
; CHECK:      cmp.l  %d0, %d1
; CHECK-SAME: encoding: [0xb2,0x80]
cmp.l	%d0, %d1
; CHECK:      cmp.l  %d7, %d1
; CHECK-SAME: encoding: [0xb2,0x87]
cmp.l	%d7, %d1
