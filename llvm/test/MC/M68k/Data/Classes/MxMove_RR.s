; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      move.b  %d0, %d1
; CHECK-SAME: encoding: [0x12,0x00]
move.b	%d0, %d1
; CHECK:      move.w  %a2, %d3
; CHECK-SAME: encoding: [0x36,0x0a]
move.w	%a2, %d3
; CHECK:      move.w  %a2, %a6
; CHECK-SAME: encoding: [0x3c,0x4a]
move.w	%a2, %a6
; CHECK:      move.w  %a2, %d1
; CHECK-SAME: encoding: [0x32,0x0a]
move.w	%a2, %d1
; CHECK:      move.l  %d2, %d1
; CHECK-SAME: encoding: [0x22,0x02]
move.l	%d2, %d1
; CHECK:      move.l  %a2, %a1
; CHECK-SAME: encoding: [0x22,0x4a]
move.l	%a2, %a1
