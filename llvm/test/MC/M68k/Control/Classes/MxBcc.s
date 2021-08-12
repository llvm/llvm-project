; RUN: llvm-mc -triple=m68k -motorola-integers -show-encoding %s | FileCheck %s

; CHECK:      bhi  $1
; CHECK-SAME: encoding: [0x62,0x01]
bhi	$1
; CHECK:      bls  $2a
; CHECK-SAME: encoding: [0x63,0x2a]
bls	$2a
; CHECK:      bcc  $1
; CHECK-SAME: encoding: [0x64,0x01]
bcc	$1
; CHECK:      bcs  $1
; CHECK-SAME: encoding: [0x65,0x01]
bcs	$1
; CHECK:      bne  $1
; CHECK-SAME: encoding: [0x66,0x01]
bne	$1
; CHECK:      beq  $1
; CHECK-SAME: encoding: [0x67,0x01]
beq	$1
; CHECK:      bvc  $1
; CHECK-SAME: encoding: [0x68,0x01]
bvc	$1
; CHECK:      bvs  $1
; CHECK-SAME: encoding: [0x69,0x01]
bvs	$1
; CHECK:      bpl  $1
; CHECK-SAME: encoding: [0x6a,0x01]
bpl	$1
; CHECK:      bmi  $1
; CHECK-SAME: encoding: [0x6b,0x01]
bmi	$1
; CHECK:      bge  $1
; CHECK-SAME: encoding: [0x6c,0x01]
bge	$1
; CHECK:      blt  $1
; CHECK-SAME: encoding: [0x6d,0x01]
blt	$1
; CHECK:      bgt  $1
; CHECK-SAME: encoding: [0x6e,0x01]
bgt	$1
; CHECK:      ble  $1
; CHECK-SAME: encoding: [0x6f,0x01]
ble	$1

; CHECK:      bhi  $3fc
; CHECK-SAME: encoding: [0x62,0x00,0x03,0xfc]
bhi	$3fc
; CHECK:      bls  $3fc
; CHECK-SAME: encoding: [0x63,0x00,0x03,0xfc]
bls	$3fc
; CHECK:      bcc  $3fc
; CHECK-SAME: encoding: [0x64,0x00,0x03,0xfc]
bcc	$3fc
; CHECK:      bcs  $3fc
; CHECK-SAME: encoding: [0x65,0x00,0x03,0xfc]
bcs	$3fc
; CHECK:      bne  $3fc
; CHECK-SAME: encoding: [0x66,0x00,0x03,0xfc]
bne	$3fc
; CHECK:      beq  $3fc
; CHECK-SAME: encoding: [0x67,0x00,0x03,0xfc]
beq	$3fc
; CHECK:      bvc  $3fc
; CHECK-SAME: encoding: [0x68,0x00,0x03,0xfc]
bvc	$3fc
; CHECK:      bvs  $3fc
; CHECK-SAME: encoding: [0x69,0x00,0x03,0xfc]
bvs	$3fc
; CHECK:      bpl  $3fc
; CHECK-SAME: encoding: [0x6a,0x00,0x03,0xfc]
bpl	$3fc
; CHECK:      bmi  $3fc
; CHECK-SAME: encoding: [0x6b,0x00,0x03,0xfc]
bmi	$3fc
; CHECK:      bge  $3fc
; CHECK-SAME: encoding: [0x6c,0x00,0x03,0xfc]
bge	$3fc
; CHECK:      blt  $3fc
; CHECK-SAME: encoding: [0x6d,0x00,0x03,0xfc]
blt	$3fc
; CHECK:      bgt  $3fc
; CHECK-SAME: encoding: [0x6e,0x00,0x03,0xfc]
bgt	$3fc
; CHECK:      ble  $3fc
; CHECK-SAME: encoding: [0x6f,0x00,0x03,0xfc]
ble	$3fc

