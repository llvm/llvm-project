@ RUN: llvm-mc %s -triple armv7-linux-gnueabi -filetype asm -o - | FileCheck %s

	.syntax unified
	.thumb

	.p2align 2
	.global emit_asm
	.type emit_asm,%function
emit_asm:
	.inst.w 0xf2400000, 0xf2c00000

@ CHECK: 	.code	16
@ CHECK: 	.p2align	2
@ CHECK: 	.globl	emit_asm
@ CHECK: 	.type	emit_asm,%function
@ CHECK: emit_asm:
@ CHECK: 	inst.w 0xf2400000
@ CHECK: 	inst.w 0xf2c00000

