; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      add.b  (0,%pc,%d1), %d0
; CHECK-SAME: encoding: [0xd0,0x3b,0x18,0x00]
add.b	(0,%pc,%d1), %d0
; CHECK:      add.b  (-1,%pc,%d1), %d0
; CHECK-SAME: encoding: [0xd0,0x3b,0x18,0xff]
add.b	(-1,%pc,%d1), %d0
; CHECK:      add.l  (0,%pc,%d1), %d0
; CHECK-SAME: encoding: [0xd0,0xbb,0x18,0x00]
add.l	(0,%pc,%d1), %d0
; CHECK:      adda.l  (0,%pc,%a2), %a1
; CHECK-SAME: encoding: [0xd3,0xfb,0xa8,0x00]
adda.l	(0,%pc,%a2), %a1

; CHECK:      add.b  (0,%pc), %d0
; CHECK-SAME: encoding: [0xd0,0x3a,0x00,0x00]
add.b	(0,%pc), %d0
; CHECK:      add.l  (-1,%pc), %d0
; CHECK-SAME: encoding: [0xd0,0xba,0xff,0xff]
add.l	(-1,%pc), %d0

; CHECK:      add.b  (0,%a0,%d1), %d0
; CHECK-SAME: encoding: [0xd0,0x30,0x18,0x00]
add.b	(0,%a0,%d1), %d0
; CHECK:      add.b  (-1,%a0,%d1), %d0
; CHECK-SAME: encoding: [0xd0,0x30,0x18,0xff]
add.b	(-1,%a0,%d1), %d0
; CHECK:      add.l  (0,%a1,%d1), %d0
; CHECK-SAME: encoding: [0xd0,0xb1,0x18,0x00]
add.l	(0,%a1,%d1), %d0
; CHECK:      adda.l  (0,%a2,%a2), %a1
; CHECK-SAME: encoding: [0xd3,0xf2,0xa8,0x00]
adda.l	(0,%a2,%a2), %a1

; CHECK:      add.b  (0,%a0), %d0
; CHECK-SAME: encoding: [0xd0,0x28,0x00,0x00]
add.b	(0,%a0), %d0
; CHECK:      add.l  (-1,%a1), %d0
; CHECK-SAME: encoding: [0xd0,0xa9,0xff,0xff]
add.l	(-1,%a1), %d0

; CHECK:      add.b  (%a0), %d0
; CHECK-SAME: encoding: [0xd0,0x10]
add.b	(%a0), %d0
; CHECK:      adda.l  (%a1), %a3
; CHECK-SAME: encoding: [0xd7,0xd1]
adda.l	(%a1), %a3

