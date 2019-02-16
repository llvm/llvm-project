## Check that the sanity check for an inconsistent header works.
## The first word contains the bad value for CodeWords, 0xf, which indicates
## that we need 0x11110 << 2 =  120 bytes of space for the unwind codes.
## It follows that the .xdata section is badly formed as only 8 bytes are
## allocated for the unwind codes.

// REQUIRES: aarch64-registered-target
// RUN: llvm-mc -filetype=obj -triple aarch64-windows %s -o - \
// RUN:   | not llvm-readobj -unwind - 2>&1 | FileCheck %s

// CHECK: LLVM ERROR: Malformed unwind data

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
	.long		0xf0800012
	.long 		0x8
	.long 		0xe
	.long 		0x100d61f
	.long 		0xe3e3e3e4

