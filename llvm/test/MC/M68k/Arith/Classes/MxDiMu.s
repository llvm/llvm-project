; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      divs  %d1, %d0
; CHECK-SAME: encoding: [0x81,0xc1]
divs	%d1, %d0
; CHECK:      divu  %d1, %d0
; CHECK-SAME: encoding: [0x80,0xc1]
divu	%d1, %d0
; CHECK:      divs  #0, %d0
; CHECK-SAME: encoding: [0x81,0xfc,0x00,0x00]
divs	#0, %d0
; CHECK:      divu  #-1, %d0
; CHECK-SAME: encoding: [0x80,0xfc,0xff,0xff]
divu	#-1, %d0
; CHECK:	divs.l	%d1, %d0
; CHECK-SAME:	encoding: [0x4c,0x41,0x08,0x00]
divs.l	%d1, %d0
; CHECK:	divu.l	%d1, %d0
; CHECK-SAME:	encoding: [0x4c,0x41,0x00,0x00]
divu.l	%d1, %d0
; CHECK:      muls  %d1, %d0
; CHECK-SAME: encoding: [0xc1,0xc1]
muls	%d1, %d0
; CHECK:      mulu  %d1, %d0
; CHECK-SAME: encoding: [0xc0,0xc1]
mulu	%d1, %d0
; CHECK:      muls  #0, %d0
; CHECK-SAME: encoding: [0xc1,0xfc,0x00,0x00]
muls	#0, %d0
; CHECK:      mulu  #-1, %d0
; CHECK-SAME: encoding: [0xc0,0xfc,0xff,0xff]
mulu	#-1, %d0
; CHECK:      muls.l  %d1, %d0
; CHECK-SAME: encoding: [0x4c,0x01,0x08,0x00]
muls.l	%d1, %d0
; CHECK:      mulu.l  %d1, %d0
; CHECK-SAME: encoding: [0x4c,0x01,0x00,0x00]
mulu.l	%d1, %d0
