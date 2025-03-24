; RUN: llvm-mc -triple=m68k -mcpu=M68010 -show-encoding %s | FileCheck %s

; CHECK:      move.w  %d1, %sr
; CHECK-SAME: encoding: [0x46,0xc1]
move.w	%d1, %sr

; CHECK:      move.w  %sr, %d1
; CHECK-SAME: encoding: [0x40,0xc1]
move.w	%sr, %d1
