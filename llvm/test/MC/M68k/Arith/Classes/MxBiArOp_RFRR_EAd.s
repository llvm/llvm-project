; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      eor.w  %d1, %d0
; CHECK-SAME: encoding: [0xb3,0x40]
eor.w	%d1, %d0
; CHECK:      eor.w  %d2, %d3
; CHECK-SAME: encoding: [0xb5,0x43]
eor.w	%d2, %d3
; CHECK:      eor.l  %d1, %d0
; CHECK-SAME: encoding: [0xb3,0x80]
eor.l	%d1, %d0
; CHECK:      eor.l  %d1, %d7
; CHECK-SAME: encoding: [0xb3,0x87]
eor.l	%d1, %d7

