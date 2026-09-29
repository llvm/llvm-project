; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      bchg  %d1, %d0
; CHECK-SAME: encoding: [0x03,0x40]
bchg	%d1, %d0
; CHECK:      bchg  %d0, %d3
; CHECK-SAME: encoding: [0x01,0x43]
bchg	%d0, %d3

; CHECK:      bclr  %d1, %d0
; CHECK-SAME: encoding: [0x03,0x80]
bclr	%d1, %d0
; CHECK:      bclr  %d0, %d3
; CHECK-SAME: encoding: [0x01,0x83]
bclr	%d0, %d3

; CHECK:      bset  %d1, %d0
; CHECK-SAME: encoding: [0x03,0xc0]
bset	%d1, %d0
; CHECK:      bset  %d0, %d3
; CHECK-SAME: encoding: [0x01,0xc3]
bset	%d0, %d3

; CHECK:      btst  %d1, %d0
; CHECK-SAME: encoding: [0x03,0x00]
btst	%d1, %d0
; CHECK:      btst  %d0, %d3
; CHECK-SAME: encoding: [0x01,0x03]
btst	%d0, %d3
