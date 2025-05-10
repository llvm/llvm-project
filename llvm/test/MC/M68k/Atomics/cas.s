; RUN: llvm-mc -show-encoding -triple=m68k -mcpu=M68020 -motorola-integers %s | FileCheck %s

; Address Register Indirect
; CHECK: cas.b %d3, %d2, (%a2)
; CHECK-SAME: ; encoding: [0x0a,0xd2,0x00,0x83]
cas.b %d3, %d2, (%a2)

; CHECK: cas.w %d4, %d5, (%a3)
; CHECK-SAME: ; encoding: [0x0c,0xd3,0x01,0x44]
cas.w %d4, %d5, (%a3)

; CHECK: cas.l %d6, %d7, (%a4)
; CHECK-SAME: ; encoding: [0x0e,0xd4,0x01,0xc6]
cas.l %d6, %d7, (%a4)

; Address Register Indirect with Displacement
; CHECK: cas.b %d3, %d2, (5,%a2)
; CHECK-SAME: ; encoding: [0x0a,0xea,0x00,0x83]
cas.b %d3, %d2, (5, %a2)

; CHECK: cas.w %d4, %d5, (6,%a3)
; CHECK-SAME: ; encoding: [0x0c,0xeb,0x01,0x44]
cas.w %d4, %d5, (6, %a3)

; CHECK: cas.l %d6, %d7, (7,%a4)
; CHECK-SAME: ; encoding: [0x0e,0xec,0x01,0xc6]
cas.l %d6, %d7, (7, %a4)

; Address Register Indirect with Index (Scale = 1)
; CHECK: cas.b %d3, %d2, (5,%a2,%d1)
; CHECK-SAME: ; encoding: [0x0a,0xf2,0x00,0x83]
cas.b %d3, %d2, (5, %a2, %d1)

; CHECK: cas.w %d4, %d5, (6,%a3,%d1)
; CHECK-SAME: ; encoding: [0x0c,0xf3,0x01,0x44]
cas.w %d4, %d5, (6, %a3, %d1)

; CHECK: cas.l %d6, %d7, (7,%a4,%d1)
; CHECK-SAME: ; encoding: [0x0e,0xf4,0x01,0xc6]
cas.l %d6, %d7, (7, %a4, %d1)

; Absolute Long Address
; CHECK: cas.b %d3, %d2, $ffffffffffffffff
; CHECK-SAME: ; encoding: [0x0a,0xf8,0x00,0x83]
cas.b %d3, %d2, $ffffffffffffffff

; CHECK: cas.w %d4, %d5, $ffffffffffffffff
; CHECK-SAME: ; encoding: [0x0c,0xf8,0x01,0x44]
cas.w %d4, %d5, $ffffffffffffffff

; CHECK: cas.l %d6, %d7, $ffffffffffffffff
; CHECK-SAME: ; encoding: [0x0e,0xf8,0x01,0xc6]
cas.l %d6, %d7, $ffffffffffffffff
