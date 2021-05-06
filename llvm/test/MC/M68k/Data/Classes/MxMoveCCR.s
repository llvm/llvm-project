; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s
	.text
	.globl	MxMoveToCCR
; CHECK-LABEL: MxMoveToCCR:
MxMoveToCCR:
	; CHECK:      move.w  %d1, %ccr
	; CHECK-SAME: encoding: [0x44,0xc1]
	move.w	%d1, %ccr

	.globl	MxMoveFromCCR
; CHECK-LABEL: MxMoveFromCCR:
MxMoveFromCCR:
	; CHECK:      move.w  %ccr, %d1
	; CHECK-SAME: encoding: [0x42,0xc1]
	move.w	%ccr, %d1

