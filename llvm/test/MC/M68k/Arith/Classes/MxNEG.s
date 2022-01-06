; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      neg.b  %d0
; CHECK-SAME: encoding: [0x44,0x00]
neg.b	%d0
; CHECK:      neg.w  %d0
; CHECK-SAME: encoding: [0x44,0x40]
neg.w	%d0
; CHECK:      neg.l  %d0
; CHECK-SAME: encoding: [0x44,0x80]
neg.l	%d0

; CHECK:      negx.b  %d0
; CHECK-SAME: encoding: [0x40,0x00]
negx.b	%d0
; CHECK:      negx.w  %d0
; CHECK-SAME: encoding: [0x40,0x40]
negx.w	%d0
; CHECK:      negx.l  %d0
; CHECK-SAME: encoding: [0x40,0x80]
negx.l	%d0
