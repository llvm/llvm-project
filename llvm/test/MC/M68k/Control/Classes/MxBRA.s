; RUN: llvm-mc -triple=m68k -motorola-integers -show-encoding %s | FileCheck %s

; CHECK:      bra  $1
; CHECK-SAME: encoding: [0x60,0x01]
bra	$1
; CHECK:      bra  $2a
; CHECK-SAME: encoding: [0x60,0x2a]
bra	$2a
; CHECK:      bra  $3fc
; CHECK-SAME: encoding: [0x60,0x00,0x03,0xfc]
bra	$3fc
