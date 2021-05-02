; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s
	.text
	.globl	MxBTST_MI_PCI
; CHECK-LABEL: MxBTST_MI_PCI:
MxBTST_MI_PCI:
	; CHECK:      btst  #0, (-1,%pc,%d1)
	; CHECK-SAME: encoding: [0x08,0x3b,0x00,0x00,0x18,0xff]
	btst	#0, (-1,%pc,%d1)
	; CHECK:      btst  #1, (0,%pc,%d0)
	; CHECK-SAME: encoding: [0x08,0x3b,0x00,0x01,0x08,0x00]
	btst	#1, (0,%pc,%d0)

	.globl	MxBTST_MI_PCD
; CHECK-LABEL: MxBTST_MI_PCD:
MxBTST_MI_PCD:
	; CHECK:      btst  #0, (0,%pc)
	; CHECK-SAME: encoding: [0x08,0x3a,0x00,0x00,0x00,0x00]
	btst	#0, (0,%pc)
	; CHECK:      btst  #-1, (-1,%pc)
	; CHECK-SAME: encoding: [0x08,0x3a,0x00,0xff,0xff,0xff]
	btst	#-1, (-1,%pc)

	.globl	MxBTST_MI_ARII
; CHECK-LABEL: MxBTST_MI_ARII:
MxBTST_MI_ARII:
	; CHECK:      btst  #0, (-1,%a1,%a0)
	; CHECK-SAME: encoding: [0x08,0x31,0x00,0x00,0x88,0xff]
	btst	#0, (-1,%a1,%a0)
	; CHECK:      btst  #-1, (0,%a0,%a0)
	; CHECK-SAME: encoding: [0x08,0x30,0x00,0xff,0x88,0x00]
	btst	#-1, (0,%a0,%a0)

	.globl	MxBTST_MI_ARID
; CHECK-LABEL: MxBTST_MI_ARID:
MxBTST_MI_ARID:
	; CHECK:      btst  #0, (-1,%a1)
	; CHECK-SAME: encoding: [0x08,0x29,0x00,0x00,0xff,0xff]
	btst	#0, (-1,%a1)
	; CHECK:      btst  #-1, (0,%a0)
	; CHECK-SAME: encoding: [0x08,0x28,0x00,0xff,0x00,0x00]
	btst	#-1, (0,%a0)

	.globl	MxBTST_MI_ARI
; CHECK-LABEL: MxBTST_MI_ARI:
MxBTST_MI_ARI:
	; CHECK:      btst  #0, (%a1)
	; CHECK-SAME: encoding: [0x08,0x11,0x00,0x00]
	btst	#0, (%a1)
	; CHECK:      btst  #-1, (%a0)
	; CHECK-SAME: encoding: [0x08,0x10,0x00,0xff]
	btst	#-1, (%a0)

