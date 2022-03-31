; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      eor.b  %d1, %d0
; CHECK-SAME: encoding: [0xb3,0x00]
eor.b	%d1, %d0
; CHECK:      eor.b  %d4, %d5
; CHECK-SAME: encoding: [0xb9,0x05]
eor.b	%d4, %d5
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
