; RUN: llvm-mc -show-encoding -triple=m68k %s | FileCheck %s

; CHECK: cas.b %d3, %d2, (%a2)
; CHECK-SAME: ; encoding: [0x0a,0xd2,0x00,0x83]
cas.b %d3, %d2, (%a2)

; CHECK: cas.w %d4, %d5, (%a3)
; CHECK-SAME: ; encoding: [0x0c,0xd3,0x01,0x44]
cas.w %d4, %d5, (%a3)

; CHECK: cas.l %d6, %d7, (%a4)
; CHECK-SAME: ; encoding: [0x0e,0xd4,0x01,0xc6]
cas.l %d6, %d7, (%a4)
