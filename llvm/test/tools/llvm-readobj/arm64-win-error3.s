## Check that error handling for going past the unwind data works.
## .xdata below contains bad opcodes in the last word.  The last byte, 0xe0,
## indicates that we have come across alloc_l, which requires 4 bytes. In this
## case, unwind code processing will go past the allocated unwind data.

// REQUIRES: aarch64-registered-target
// RUN: llvm-mc -filetype=obj -triple aarch64-windows %s -o - \
// RUN:   | llvm-readobj -unwind - | FileCheck %s

// CHECK: Prologue [
// CHECK:   Opcode 0xe0 goes past the unwind data

	.text
	.globl	"?func@@YAHXZ"
	.p2align	3
"?func@@YAHXZ":
	sub     sp,sp,#0x10
	stp     x19,lr,[sp]
	sub     sp,sp,#0x1F0
	mov     w19,w0
	bl	"?func2@@YAXXZ"
	cmp     w19,#2
	ble     .LBB0_1
	bl      "?func2@@YAHXZ"
	add      sp,sp,#0x1F0
	ldp      x19,lr,[sp]
	add      sp,sp,#0x10
	ret
.LBB0_1:
	mov      x0,sp
	bl       "?func3@@YAHPEAH@Z"
	add      sp,sp,#0x1F0
	ldp      x19,lr,[sp]
	add      sp,sp,#0x10
	ret


.section .pdata,"dr"
	.long "?func@@YAHXZ"@IMGREL
        .long "$unwind$func@@YAHXZ"@IMGREL


.section	.xdata,"dr"
"$unwind$func@@YAHXZ":
        .p2align	3
	.long		0x10800012
	.long 		0x8
	.long 		0xe
	.long 		0x100d61f
	.long 		0xe0000000

