; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

	; CHECK:      beq  .LBB0_1
	; CHECK-SAME: encoding: [0x67,A]
        ; CHECK:      fixup A - offset: 1, value: .LBB0_1-1, kind: FK_PCRel_1
	beq	.LBB0_1
	; CHECK:      bra  .LBB0_2
	; CHECK-SAME: encoding: [0x60,A]
        ; CHECK:      fixup A - offset: 1, value: .LBB0_2-1, kind: FK_PCRel_1
	bra	.LBB0_2
.LBB0_1:
	; CHECK:      add.l  #0, %d0
	; CHECK-SAME: encoding: [0xd0,0xbc,0x00,0x00,0x00,0x00]
	add.l	#0, %d0
	; CHECK:      rts
	; CHECK-SAME: encoding: [0x4e,0x75]
	rts
.LBB0_2:
	; CHECK:      add.l  #1, %d0
	; CHECK-SAME: encoding: [0xd0,0xbc,0x00,0x00,0x00,0x01]
	add.l	#1, %d0
	; CHECK:      rts
	; CHECK-SAME: encoding: [0x4e,0x75]
	rts
