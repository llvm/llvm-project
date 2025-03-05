; RUN: llvm-mc -triple=m68k -motorola-integers -filetype=obj --position-independent < %s \
; RUN:     | llvm-objdump -d - | FileCheck %s

; CHECK-LABEL: <TIGHT>:
TIGHT:
  ; CHECK: bsr.w   $7a
	bsr.w	.LBB0_2
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
  ; CHECK: bsr.b   $82
	bsr.b	.LBB1_2
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
  ; CHECK: bsr.w    $2
	bsr.w	.LBB2_1
.LBB2_1:
	add.l	#0, %d0
	rts
