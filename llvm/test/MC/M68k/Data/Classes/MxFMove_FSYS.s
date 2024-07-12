; RUN: llvm-mc -triple=m68k -mcpu=M68040 -show-encoding < %s | FileCheck %s

; CHECK: fmove.l %d0, %fpcr
; CHECK-SAME: [0xf2,0x00,0x90,0x00]
fmove.l %d0, %fpc

; CHECK: fmove.l %fpsr, %d2
; CHECK-SAME: [0xf2,0x02,0xa8,0x00]
fmove.l %fps, %d2

; CHECK: fmove.l %fpiar, %a3
; CHECK-SAME: [0xf2,0x0b,0xa4,0x00]
fmove.l %fpiar, %a3

; CHECK: fmove.l %a1, %fpiar
; CHECK-SAME: [0xf2,0x09,0x84,0x00]
fmove.l %a1, %fpi
