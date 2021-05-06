; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      add.b  %d0, (0,%a0,%d1)
; CHECK-SAME: encoding: [0xd1,0x30,0x18,0x00]
add.b	%d0, (0,%a0,%d1)
; CHECK:      add.b  %d0, (-1,%a0,%d1)
; CHECK-SAME: encoding: [0xd1,0x30,0x18,0xff]
add.b	%d0, (-1,%a0,%d1)
; CHECK:      add.l  %d0, (0,%a1,%d1)
; CHECK-SAME: encoding: [0xd1,0xb1,0x18,0x00]
add.l	%d0, (0,%a1,%d1)
; CHECK:      add.l  %d1, (0,%a2,%a2)
; CHECK-SAME: encoding: [0xd3,0xb2,0xa8,0x00]
add.l	%d1, (0,%a2,%a2)

; CHECK:      add.b  %d0, (0,%a0)
; CHECK-SAME: encoding: [0xd1,0x28,0x00,0x00]
add.b	%d0, (0,%a0)
; CHECK:      add.l  %d0, (-1,%a1)
; CHECK-SAME: encoding: [0xd1,0xa9,0xff,0xff]
add.l	%d0, (-1,%a1)

; CHECK:      add.b  %d0, (%a0)
; CHECK-SAME: encoding: [0xd1,0x10]
add.b	%d0, (%a0)
; CHECK:      add.l  %d3, (%a1)
; CHECK-SAME: encoding: [0xd7,0x91]
add.l	%d3, (%a1)

