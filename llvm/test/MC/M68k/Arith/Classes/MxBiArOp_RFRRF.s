; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      addx.w  %d1, %d0
; CHECK-SAME: encoding: [0xd1,0x41]
addx.w	%d1, %d0
; CHECK:      addx.w  %d2, %d3
; CHECK-SAME: encoding: [0xd7,0x42]
addx.w	%d2, %d3
; CHECK:      addx.l  %d1, %d0
; CHECK-SAME: encoding: [0xd1,0x81]
addx.l	%d1, %d0
; CHECK:      addx.l  %d1, %d7
; CHECK-SAME: encoding: [0xdf,0x81]
addx.l	%d1, %d7

