; RUN: llvm-mc -triple=m68k -motorola-integers -filetype=obj < %s \
; RUN:     | llvm-objdump -d - | FileCheck %s

; CHECK-LABEL: <TIGHT>:
TIGHT:
        ; CHECK: bra  $78
	bra	.LBB0_2
	move.l	$0, $0
	move.l	$0, $0
	move.l	$0, $0
	move.l	$0, $0
	move.l	$0, $0
	move.l	$0, $0
	move.l	$0, $0
	move.l	$0, $0
	move.l	$0, $0
	move.l	$0, $0
	move.l	$0, $0
	move.l	$0, $0
.LBB0_2:
	add.l	#0, %d0
	rts

; CHECK-LABEL: <RELAXED>:
RELAXED:
        ; CHECK: bra  $84
	bra	.LBB1_2
	move.l	$0, $0
	move.l	$0, $0
	move.l	$0, $0
	move.l	$0, $0
	move.l	$0, $0
	move.l	$0, $0
	move.l	$0, $0
	move.l	$0, $0
	move.l	$0, $0
	move.l	$0, $0
	move.l	$0, $0
	move.l	$0, $0
	move.l	$0, $0
.LBB1_2:
	add.l	#0, %d0
	rts

; CHECK-LABEL: <ZERO>:
ZERO:
        ; CHECK: bra  $2
	bra	.LBB2_1
.LBB2_1:
	add.l	#0, %d0
	rts
