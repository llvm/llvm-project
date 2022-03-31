; RUN: llvm-mc -triple=m68k -show-encoding %s | FileCheck %s

        ; A fixup whose size is multiple of a word.
	; CHECK:      cmpi.l  #87, (.LBB0_1,%pc)
	; CHECK-SAME: encoding: [0x0c,0xba,0x00,0x00,0x00,0x57,A,A]
        ; CHECK:      fixup A - offset: 6, value: .LBB0_1, kind: FK_PCRel_2
	cmpi.l  #87, (.LBB0_1,%pc)

        ; A fixup that is smaller than a word.
        ; For cases where the fixup is located in the first word, they are
        ; tested by `Control/branch-pc-rel.s`.
	; CHECK:      cmpi.l  #94, (.LBB0_2,%pc,%a0)
	; CHECK-SAME: encoding: [0x0c,0xbb,0x00,0x00,0x00,0x5e,0x88,A]
        ; CHECK:      fixup A - offset: 7, value: .LBB0_2+1, kind: FK_PCRel_1
	cmpi.l  #94, (.LBB0_2,%pc,%a0)
.LBB0_1:
	add.l	#0, %d0
	rts
.LBB0_2:
	add.l	#1, %d0
	rts

