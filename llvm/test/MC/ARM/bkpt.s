@ RUN: llvm-mc -triple armv7-unknown-unknown -filetype asm -o - %s | FileCheck %s

	.syntax unified
	.thumb

	.global thumb_default_bkpt
	.type thumb_default_bkpt, %function
	.thumb_func
thumb_default_bkpt:
	bkpt

@ CHECK-LABEL: thumb_default_bkpt
@ CHECK: bkpt #0

	.global normal_bkpt
	.type normal_bkpt, %function
normal_bkpt:
	bkpt #42

@ CHECK-LABEL: normal_bkpt
@ CHECK: bkpt #42

	.arm

	.global arm_default_bkpt
	.type arm_default_bkpt, %function
arm_default_bkpt:
	bkpt

@ CHECK-LABEL: arm_default_bkpt
@ CHECK: bkpt #0

