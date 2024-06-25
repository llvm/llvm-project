; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      move.b  #-1, %d0
; CHECK-SAME: encoding: [0x10,0x3c,0x00,0xff]
move.b	#-1, %d0
; CHECK:      move.l  #42, %a1
; CHECK-SAME: encoding: [0x22,0x7c,0x00,0x00,0x00,0x2a]
move.l	#42, %a1
; CHECK:      move.l  #-1, %a1
; CHECK-SAME: encoding: [0x22,0x7c,0xff,0xff,0xff,0xff]
move.l	#-1, %a1
; CHECK:      moveq  #-17, %d3
; CHECK-SAME: encoding: [0x76,0xef]
moveq	#-17, %d3
