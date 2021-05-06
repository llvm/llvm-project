; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s
	.text
	.globl	MxLEA_PCD
; CHECK-LABEL: MxLEA_PCD:
MxLEA_PCD:
	; CHECK:      lea  (0,%pc), %a0
	; CHECK-SAME: encoding: [0x41,0xfa,0x00,0x00]
	lea	(0,%pc), %a0
	; CHECK:      lea  (-1,%pc), %a0
	; CHECK-SAME: encoding: [0x41,0xfa,0xff,0xff]
	lea	(-1,%pc), %a0

	.globl	MxLEA_ARII
; CHECK-LABEL: MxLEA_ARII:
MxLEA_ARII:
	; CHECK:      lea  (0,%a1,%d1), %a0
	; CHECK-SAME: encoding: [0x41,0xf1,0x18,0x00]
	lea	(0,%a1,%d1), %a0
	; CHECK:      lea  (0,%a2,%a2), %a1
	; CHECK-SAME: encoding: [0x43,0xf2,0xa8,0x00]
	lea	(0,%a2,%a2), %a1

	.globl	MxLEA_ARID
; CHECK-LABEL: MxLEA_ARID:
MxLEA_ARID:
	; CHECK:      lea  (-1,%a1), %a0
	; CHECK-SAME: encoding: [0x41,0xe9,0xff,0xff]
	lea	(-1,%a1), %a0
	; CHECK:      lea  (-1,%a1), %a0
	; CHECK-SAME: encoding: [0x41,0xe9,0xff,0xff]
	lea	(-1,%a1), %a0

