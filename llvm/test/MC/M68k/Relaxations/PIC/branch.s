; RUN: llvm-mc -triple=m68k -motorola-integers -filetype=obj --position-independent < %s \
; RUN:     | llvm-objdump -d - | FileCheck %s
; RUN: llvm-mc -triple m68k -show-encoding --position-independent %s -o - \
; RUN:   | FileCheck -check-prefix=INSTR -check-prefix=FIXUP %s

; CHECK-LABEL: <TIGHT>:
TIGHT:
	; CHECK: bra  $7f
	; INSTR: bra .LBB0_2 ; encoding: [0x60,A]
	; FIXUP: fixup A - offset: 1, value: .LBB0_2-1, kind: FK_PCRel_1
	bra	.LBB0_2
	.space 0x7F  ; i8::MAX
.LBB0_2:
	add.l	#0, %d0
	rts

; CHECK-LABEL: <RELAXED>:
RELAXED:
	; CHECK: bra  $82
	; INSTR: bra .LBB1_2 ; encoding: [0x60,A]
	; FIXUP: fixup A - offset: 1, value: .LBB1_2-1, kind: FK_PCRel_1
	bra	.LBB1_2
	.space 0x80  ; Greater than i8::MAX
.LBB1_2:
	add.l	#0, %d0
	rts

; CHECK-LABEL: <ZERO>:
ZERO:
	; CHECK: bra  $2
	; INSTR: bra .LBB3_1 ; encoding: [0x60,A]
	; FIXUP: fixup A - offset: 1, value: .LBB3_1-1, kind: FK_PCRel_1
	bra	.LBB3_1
.LBB3_1:
	add.l	#0, %d0
	rts


