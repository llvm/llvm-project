; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      btst  #1, %d0
; CHECK-SAME: encoding: [0x08,0x00,0x00,0x01]
btst	#1, %d0
; CHECK:      btst  #0, %d3
; CHECK-SAME: encoding: [0x08,0x03,0x00,0x00]
btst	#0, %d3

