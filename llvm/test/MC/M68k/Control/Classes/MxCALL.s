; RUN: llvm-mc -triple=m68k -motorola-integers -show-encoding %s | FileCheck %s

; CHECK:      jsr  (0,%pc,%a0)
; CHECK-SAME: encoding: [0x4e,0xbb,0x88,0x00]
jsr	(0,%pc,%a0)
; CHECK:      jsr  (-1,%pc,%a0)
; CHECK-SAME: encoding: [0x4e,0xbb,0x88,0xff]
jsr	(-1,%pc,%a0)
; CHECK:      jsr  (42,%pc,%a0)
; CHECK-SAME: encoding: [0x4e,0xbb,0x88,0x2a]
jsr	(42,%pc,%a0)

; CHECK:      jsr  (0,%pc)
; CHECK-SAME: encoding: [0x4e,0xba,0x00,0x00]
jsr	(0,%pc)
; CHECK:      jsr  (32767,%pc)
; CHECK-SAME: encoding: [0x4e,0xba,0x7f,0xff]
jsr	(32767,%pc)

; CHECK:      jsr  $2a
; CHECK-SAME: encoding: [0x4e,0xb9,0x00,0x00,0x00,0x2a]
jsr	$2a
; CHECK:      jsr  $ffffffffffffffff
; CHECK-SAME: encoding: [0x4e,0xb9,0xff,0xff,0xff,0xff]
jsr	$ffffffffffffffff

; CHECK:      jsr  (%a0)
; CHECK-SAME: encoding: [0x4e,0x90]
jsr	(%a0)
; CHECK:      jsr  (%a1)
; CHECK-SAME: encoding: [0x4e,0x91]
jsr	(%a1)
; CHECK:      jsr  (%a2)
; CHECK-SAME: encoding: [0x4e,0x92]
jsr	(%a2)

