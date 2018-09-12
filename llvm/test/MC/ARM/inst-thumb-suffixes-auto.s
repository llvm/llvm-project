@ RUN: llvm-mc %s -triple armv7-linux-gnueabi -filetype asm -o - \
@ RUN:   | FileCheck %s
@ RUN: llvm-mc %s -triple armebv7-linux-gnueabi -filetype asm -o - \
@ RUN:   | FileCheck %s

	.syntax unified
	.thumb

	.align 2
	.global inst_n
	.type inst_n,%function
inst_n:
	@ bx lr, mov.w r0, #42
	.inst 0x4770, 0xf04f002a
@ CHECK: .inst.n 0x4770
@ CHECK: .inst.w 0xf04f002a
