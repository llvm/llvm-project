; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      move.b  %d0, (0,%a0,%d1)
; CHECK-SAME: encoding: [0x11,0x80,0x18,0x00]
move.b	%d0, (0,%a0,%d1)
; CHECK:      move.b  %d0, (-1,%a0,%d1)
; CHECK-SAME: encoding: [0x11,0x80,0x18,0xff]
move.b	%d0, (-1,%a0,%d1)
; CHECK:      move.l  %d0, (0,%a1,%d1)
; CHECK-SAME: encoding: [0x23,0x80,0x18,0x00]
move.l	%d0, (0,%a1,%d1)
; CHECK:      move.l  %d1, (0,%a2,%a2)
; CHECK-SAME: encoding: [0x25,0x81,0xa8,0x00]
move.l	%d1, (0,%a2,%a2)

; CHECK:      move.b  %d0, (0,%a0)
; CHECK-SAME: encoding: [0x11,0x40,0x00,0x00]
move.b	%d0, (0,%a0)
; CHECK:      move.l  %d0, (-1,%a1)
; CHECK-SAME: encoding: [0x23,0x40,0xff,0xff]
move.l	%d0, (-1,%a1)
; CHECK:      move.l  %a0, (-1,%a1)
; CHECK-SAME: encoding: [0x23,0x48,0xff,0xff]
move.l	%a0, (-1,%a1)

; CHECK:      move.b  %d0, (%a0)
; CHECK-SAME: encoding: [0x10,0x80]
move.b	%d0, (%a0)
; CHECK:      move.l  %d3, (%a1)
; CHECK-SAME: encoding: [0x22,0x83]
move.l	%d3, (%a1)
; CHECK:      move.l  %a4, (%a1)
; CHECK-SAME: encoding: [0x22,0x8c]
move.l	%a4, (%a1)
