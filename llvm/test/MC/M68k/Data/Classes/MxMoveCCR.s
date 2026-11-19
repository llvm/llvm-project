; RUN: llvm-mc -triple=m68k -mcpu=M68010 -show-encoding %s | FileCheck %s

; CHECK:      move.w  %d1, %ccr
; CHECK-SAME: encoding: [0x44,0xc1]
move.w	%d1, %ccr

; CHECK:      move.w  #1235, %ccr
; CHECK-SAME: encoding: [0x44,0xfc,0x04,0xd3]
move.w	#1235, %ccr

; CHECK:      move.w  %ccr, %d1
; CHECK-SAME: encoding: [0x42,0xc1]
move.w	%ccr, %d1
