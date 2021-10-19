; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s
	.text
	.globl	MxBiArOp_RFRR_xEA
; CHECK-LABEL: MxBiArOp_RFRR_xEA:
MxBiArOp_RFRR_xEA:
	; CHECK:      add.w  %d1, %d0
	; CHECK-SAME: encoding: [0xd0,0x41]
	add.w	%d1, %d0
	; CHECK:      add.w  %d2, %d3
	; CHECK-SAME: encoding: [0xd6,0x42]
	add.w	%d2, %d3
	; CHECK:      add.l  %d1, %d0
	; CHECK-SAME: encoding: [0xd0,0x81]
	add.l	%d1, %d0
	; CHECK:      add.l  %a1, %d0
	; CHECK-SAME: encoding: [0xd0,0x89]
	add.l	%a1, %d0
	; CHECK:      add.l  %a1, %d7
	; CHECK-SAME: encoding: [0xde,0x89]
	add.l	%a1, %d7
	; CHECK:      adda.l  %d1, %a0
	; CHECK-SAME: encoding: [0xd1,0xc1]
	adda.l	%d1, %a0

