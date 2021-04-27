; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      eori.w  #0, %d0
; CHECK-SAME: encoding: [0x0a,0x40,0x00,0x00]
eori.w	#0, %d0
; CHECK:      eori.w  #-1, %d3
; CHECK-SAME: encoding: [0x0a,0x43,0xff,0xff]
eori.w	#-1, %d3
; CHECK:      eori.l  #-1, %d0
; CHECK-SAME: encoding: [0x0a,0x80,0xff,0xff,0xff,0xff]
eori.l	#-1, %d0
; CHECK:      eori.l  #131071, %d0
; CHECK-SAME: encoding: [0x0a,0x80,0x00,0x01,0xff,0xff]
eori.l	#131071, %d0
; CHECK:      eori.l  #458752, %d7
; CHECK-SAME: encoding: [0x0a,0x87,0x00,0x07,0x00,0x00]
eori.l	#458752, %d7

