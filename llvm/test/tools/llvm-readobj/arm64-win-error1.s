## Check that error handling for bad opcodes works.
## .xdata below contains the bad opcode 0xdf in the 4th word of .xdata.

// REQUIRES: aarch64-registered-target
// RUN: llvm-mc -filetype=obj -triple aarch64-windows %s -o - \
// RUN:   | llvm-readobj -unwind - | FileCheck %s

// CHECK:     Prologue [
// CHECK:        0xdf                ; Bad opcode!
// CHECK:        0xff                ; Bad opcode!
// CHECK:        0xd600              ; stp x19, lr, [sp, #0]
// CHECK:        0x01                ; sub sp, #16
// CHECK:        0xe4                ; end
// CHECK:     ]

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
	.long 		0x00d6ffdf
	.long 		0xe3e3e401

