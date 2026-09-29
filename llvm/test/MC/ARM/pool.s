@ RUN: llvm-mc -triple thumbv7-linux-gnueabi -o - %s | FileCheck %s

	.syntax unified

	.p2align 2
	.global pool
	.type pool,%function
pool:
	ldr r0, =0xba5eba11
	bx lr
	.pool

@ CHECK-LABEL: pool
@ CHECK: ldr r0, .Ltmp0
@ CHECK: .p2align	2
@ CHECK-LABEL: .Ltmp0:
@ CHECK: .long	3126770193


