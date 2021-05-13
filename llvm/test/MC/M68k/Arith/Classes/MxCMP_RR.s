; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      cmp.b  %d0, %d1
; CHECK-SAME: encoding: [0xb2,0x00]
cmp.b	%d0, %d1
; CHECK:      cmp.b  %d3, %d2
; CHECK-SAME: encoding: [0xb4,0x03]
cmp.b	%d3, %d2
; CHECK:      cmp.l  %d0, %d1
; CHECK-SAME: encoding: [0xb2,0x80]
cmp.l	%d0, %d1
; CHECK:      cmp.l  %d7, %d1
; CHECK-SAME: encoding: [0xb2,0x87]
cmp.l	%d7, %d1

