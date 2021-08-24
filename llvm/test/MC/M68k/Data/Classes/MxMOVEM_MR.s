; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      movem.l  %d0, (0,%a1)
; CHECK-SAME: encoding: [0x48,0xe9,0x00,0x01,0x00,0x00]
movem.l	%d0, (0,%a1)
; CHECK:      movem.l  %d0-%d1, (-1,%a1)
; CHECK-SAME: encoding: [0x48,0xe9,0x00,0x03,0xff,0xff]
movem.l	%d0-%d1, (-1,%a1)

; CHECK:      movem.l  %d0, (%a1)
; CHECK-SAME: encoding: [0x48,0xd1,0x00,0x01]
movem.l	%d0, (%a1)
; CHECK:      movem.l  %d0-%d1, (%a1)
; CHECK-SAME: encoding: [0x48,0xd1,0x00,0x03]
movem.l	%d0-%d1, (%a1)

