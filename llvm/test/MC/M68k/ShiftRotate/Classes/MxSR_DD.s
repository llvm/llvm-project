; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      lsl.b  %d0, %d1
; CHECK-SAME: encoding: [0xe1,0x29]
lsl.b	%d0, %d1
; CHECK:      lsl.l  %d1, %d2
; CHECK-SAME: encoding: [0xe3,0xaa]
lsl.l	%d1, %d2
; CHECK:      lsr.b  %d2, %d3
; CHECK-SAME: encoding: [0xe4,0x2b]
lsr.b	%d2, %d3
; CHECK:      lsr.l  %d3, %d4
; CHECK-SAME: encoding: [0xe6,0xac]
lsr.l	%d3, %d4
; CHECK:      asr.b  %d4, %d5
; CHECK-SAME: encoding: [0xe8,0x25]
asr.b	%d4, %d5
; CHECK:      asr.l  %d5, %d6
; CHECK-SAME: encoding: [0xea,0xa6]
asr.l	%d5, %d6
; CHECK:      rol.b  %d6, %d7
; CHECK-SAME: encoding: [0xed,0x3f]
rol.b	%d6, %d7
; CHECK:      rol.l  %d7, %d1
; CHECK-SAME: encoding: [0xef,0xb9]
rol.l	%d7, %d1
; CHECK:      ror.b  %d0, %d1
; CHECK-SAME: encoding: [0xe0,0x39]
ror.b	%d0, %d1
; CHECK:      ror.l  %d0, %d1
; CHECK-SAME: encoding: [0xe0,0xb9]
ror.l	%d0, %d1

