; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      lsl.b  #1, %d1
; CHECK-SAME: encoding: [0xe3,0x09]
lsl.b	#1, %d1
; CHECK:      lsl.l  #1, %d1
; CHECK-SAME: encoding: [0xe3,0x89]
lsl.l	#1, %d1
; CHECK:      lsr.b  #1, %d1
; CHECK-SAME: encoding: [0xe2,0x09]
lsr.b	#1, %d1
; CHECK:      lsr.l  #1, %d1
; CHECK-SAME: encoding: [0xe2,0x89]
lsr.l	#1, %d1
; CHECK:      asr.b  #1, %d1
; CHECK-SAME: encoding: [0xe2,0x01]
asr.b	#1, %d1
; CHECK:      asr.l  #1, %d1
; CHECK-SAME: encoding: [0xe2,0x81]
asr.l	#1, %d1
; CHECK:      rol.b  #1, %d1
; CHECK-SAME: encoding: [0xe3,0x19]
rol.b	#1, %d1
; CHECK:      rol.l  #1, %d1
; CHECK-SAME: encoding: [0xe3,0x99]
rol.l	#1, %d1
; CHECK:      ror.b  #1, %d1
; CHECK-SAME: encoding: [0xe2,0x19]
ror.b	#1, %d1
; CHECK:      ror.l  #1, %d1
; CHECK-SAME: encoding: [0xe2,0x99]
ror.l	#1, %d1
; CHECK:      ror.l  #2, %d1
; CHECK-SAME: encoding: [0xe4,0x99]
ror.l	#2, %d1
; CHECK:      ror.l  #3, %d1
; CHECK-SAME: encoding: [0xe6,0x99]
ror.l	#3, %d1
; CHECK:      ror.l  #4, %d1
; CHECK-SAME: encoding: [0xe8,0x99]
ror.l	#4, %d1
; CHECK:      ror.l  #5, %d1
; CHECK-SAME: encoding: [0xea,0x99]
ror.l	#5, %d1
; CHECK:      ror.l  #6, %d1
; CHECK-SAME: encoding: [0xec,0x99]
ror.l	#6, %d1
; CHECK:      ror.l  #7, %d1
; CHECK-SAME: encoding: [0xee,0x99]
ror.l	#7, %d1
; CHECK:      ror.l  #8, %d1
; CHECK-SAME: encoding: [0xe0,0x99]
ror.l	#8, %d1

