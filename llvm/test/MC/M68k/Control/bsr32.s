; RUN: llvm-mc -triple=m68k --mcpu=M68020 -show-encoding %s | FileCheck %s

	; CHECK:      bsr.b   .LBB0_1
	; CHECK-SAME: encoding: [0x61,A]
        ; CHECK:      fixup A - offset: 1, value: .LBB0_1-1, kind: FK_PCRel_1
	bsr.b .LBB0_1
	; CHECK:      bsr.w   .LBB0_2
	; CHECK-SAME: encoding: [0x61,0x00,A,A]
        ; CHECK:      fixup A - offset: 2, value: .LBB0_2, kind: FK_PCRel_2
	bsr.w	.LBB0_2
  ; CHECK:     bsr.l   .LBB0_3
  ; CHECK-SAME: encoding: [0x61,0xff,A,A,A,A] 
        ; CHECK:      fixup A - offset: 2, value: .LBB0_3, kind: FK_PCRel_4
  bsr.l .LBB0_3  
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
.LBB0_3:
	; CHECK:      add.l  #1, %d0
	; CHECK-SAME: encoding: [0xd0,0xbc,0x00,0x00,0x00,0x01]
	add.l	#1, %d0
	; CHECK:      rts
	; CHECK-SAME: encoding: [0x4e,0x75]
	rts
