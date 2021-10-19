; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      cmpi.b  #0, %d1
; CHECK-SAME: encoding: [0x0c,0x01,0x00,0x00]
cmpi.b	#0, %d1
; CHECK:      cmpi.b  #-1, %d0
; CHECK-SAME: encoding: [0x0c,0x00,0x00,0xff]
cmpi.b	#-1, %d0
; CHECK:      cmpi.l  #13, %d7
; CHECK-SAME: encoding: [0x0c,0x87,0x00,0x00,0x00,0x0d]
cmpi.l	#13, %d7
; CHECK:      cmpi.l  #42, %d1
; CHECK-SAME: encoding: [0x0c,0x81,0x00,0x00,0x00,0x2a]
cmpi.l	#42, %d1

