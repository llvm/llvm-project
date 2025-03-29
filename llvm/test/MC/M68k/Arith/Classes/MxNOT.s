; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      not.b  %d0
; CHECK-SAME: encoding: [0x46,0x00]
not.b	%d0
; CHECK:      not.w  %d0
; CHECK-SAME: encoding: [0x46,0x40]
not.w	%d0
; CHECK:      not.l  %d0
; CHECK-SAME: encoding: [0x46,0x80]
not.l	%d0
