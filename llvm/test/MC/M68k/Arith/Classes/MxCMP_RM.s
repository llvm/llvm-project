; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

; CHECK:      cmp.b  (0,%pc,%d1), %d0
; CHECK-SAME: encoding: [0xb0,0x3b,0x18,0x00]
cmp.b	(0,%pc,%d1), %d0
; CHECK:      cmp.b  (-1,%pc,%d1), %d0
; CHECK-SAME: encoding: [0xb0,0x3b,0x18,0xff]
cmp.b	(-1,%pc,%d1), %d0
; CHECK:      cmp.w  (0,%pc,%d1), %d0
; CHECK-SAME: encoding: [0xb0,0x7b,0x18,0x00]
cmp.w	(0,%pc,%d1), %d0
; CHECK:      cmp.w  (-1,%pc,%d1), %d0
; CHECK-SAME: encoding: [0xb0,0x7b,0x18,0xff]
cmp.w	(-1,%pc,%d1), %d0
; CHECK:      cmp.l  (0,%pc,%d1), %d0
; CHECK-SAME: encoding: [0xb0,0xbb,0x18,0x00]
cmp.l	(0,%pc,%d1), %d0
; CHECK:      cmp.l  (-1,%pc,%d1), %d0
; CHECK-SAME: encoding: [0xb0,0xbb,0x18,0xff]
cmp.l	(-1,%pc,%d1), %d0

; CHECK:      cmp.b  (0,%pc), %d0
; CHECK-SAME: encoding: [0xb0,0x3a,0x00,0x00]
cmp.b	(0,%pc), %d0
; CHECK:      cmp.b  (-1,%pc), %d0
; CHECK-SAME: encoding: [0xb0,0x3a,0xff,0xff]
cmp.b	(-1,%pc), %d0
; CHECK:      cmp.w  (0,%pc), %d0
; CHECK-SAME: encoding: [0xb0,0x7a,0x00,0x00]
cmp.w	(0,%pc), %d0
; CHECK:      cmp.w  (-1,%pc), %d0
; CHECK-SAME: encoding: [0xb0,0x7a,0xff,0xff]
cmp.w	(-1,%pc), %d0
; CHECK:      cmp.l  (0,%pc), %d0
; CHECK-SAME: encoding: [0xb0,0xba,0x00,0x00]
cmp.l	(0,%pc), %d0
; CHECK:      cmp.l  (-1,%pc), %d0
; CHECK-SAME: encoding: [0xb0,0xba,0xff,0xff]
cmp.l	(-1,%pc), %d0

; CHECK:      cmp.b  (0,%a0,%d1), %d0
; CHECK-SAME: encoding: [0xb0,0x30,0x18,0x00]
cmp.b	(0,%a0,%d1), %d0
; CHECK:      cmp.b  (-1,%a0,%d1), %d0
; CHECK-SAME: encoding: [0xb0,0x30,0x18,0xff]
cmp.b	(-1,%a0,%d1), %d0
; CHECK:      cmp.w  (0,%a3,%d2), %d1
; CHECK-SAME: encoding: [0xb2,0x73,0x28,0x00]
cmp.w	(0,%a3,%d2), %d1
; CHECK:      cmp.w  (-1,%a4,%d1), %d0
; CHECK-SAME: encoding: [0xb0,0x74,0x18,0xff]
cmp.w	(-1,%a4,%d1), %d0
; CHECK:      cmp.l  (0,%a1,%d1), %d0
; CHECK-SAME: encoding: [0xb0,0xb1,0x18,0x00]
cmp.l	(0,%a1,%d1), %d0
; CHECK:      cmp.l  (0,%a2,%a2), %d1
; CHECK-SAME: encoding: [0xb2,0xb2,0xa8,0x00]
cmp.l	(0,%a2,%a2), %d1

; CHECK:      cmp.b  (0,%a0), %d0
; CHECK-SAME: encoding: [0xb0,0x28,0x00,0x00]
cmp.b	(0,%a0), %d0
; CHECK:      cmp.b  (-1,%a1), %d0
; CHECK-SAME: encoding: [0xb0,0x29,0xff,0xff]
cmp.b	(-1,%a1), %d0
; CHECK:      cmp.w  (0,%a0), %d0
; CHECK-SAME: encoding: [0xb0,0x68,0x00,0x00]
cmp.w	(0,%a0), %d0
; CHECK:      cmp.w  (-1,%a1), %d0
; CHECK-SAME: encoding: [0xb0,0x69,0xff,0xff]
cmp.w	(-1,%a1), %d0
; CHECK:      cmp.l  (0,%a0), %d0
; CHECK-SAME: encoding: [0xb0,0xa8,0x00,0x00]
cmp.l	(0,%a0), %d0
; CHECK:      cmp.l  (-1,%a1), %d0
; CHECK-SAME: encoding: [0xb0,0xa9,0xff,0xff]
cmp.l	(-1,%a1), %d0

; CHECK:      cmp.b  (%a0), %d0
; CHECK-SAME: encoding: [0xb0,0x10]
cmp.b	(%a0), %d0
; CHECK:      cmp.b  (%a0), %d1
; CHECK-SAME: encoding: [0xb2,0x10]
cmp.b	(%a0), %d1
; CHECK:      cmp.w  (%a1), %d0
; CHECK-SAME: encoding: [0xb0,0x51]
cmp.w	(%a1), %d0
; CHECK:      cmp.w  (%a1), %d1
; CHECK-SAME: encoding: [0xb2,0x51]
cmp.w	(%a1), %d1
; CHECK:      cmp.l  (%a1), %d2
; CHECK-SAME: encoding: [0xb4,0x91]
cmp.l	(%a1), %d2
; CHECK:      cmp.l  (%a1), %d3
; CHECK-SAME: encoding: [0xb6,0x91]
cmp.l	(%a1), %d3
