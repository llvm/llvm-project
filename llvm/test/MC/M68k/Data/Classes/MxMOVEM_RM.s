; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      movem.l  (0,%a1), %d0
; CHECK-SAME: encoding: [0x4c,0xe9,0x00,0x01,0x00,0x00]
movem.l	(0,%a1), %d0
; CHECK:      movem.l  (-1,%a1), %d0-%d1
; CHECK-SAME: encoding: [0x4c,0xe9,0x00,0x03,0xff,0xff]
movem.l	(-1,%a1), %d0-%d1

; CHECK:      movem.l  (%a1), %d0
; CHECK-SAME: encoding: [0x4c,0xd1,0x00,0x01]
movem.l	(%a1), %d0
; CHECK:      movem.l  (%a1), %d0-%d1
; CHECK-SAME: encoding: [0x4c,0xd1,0x00,0x03]
movem.l	(%a1), %d0-%d1
