@ RUN: llvm-mc -triple thumbv8m.base-eabi -filetype asm -o - %s | FileCheck %s
@ RUN: llvm-mc -triple thumbv8m.base-eabi -filetype obj -o - %s | llvm-readobj -r - \
@ RUN:   | FileCheck -check-prefix CHECK-RELOCATIONS %s

	.syntax unified

	.type function,%function
function:
	bx lr

	.global external
	.type external,%function

	.type test,%function
test:
	movw r0, :lower16:function
	movt r0, :upper16:function

@ CHECK-LABEL: test:
@ CHECK: 	movw r0, :lower16:function
@ CHECK: 	movt r0, :upper16:function

@ CHECK-RELOCATIONS: Relocations [
@ CHECK-RELOCATIONS:   0x2 R_ARM_THM_MOVW_ABS_NC function
@ CHECK-RELOCATIONS:   0x6 R_ARM_THM_MOVT_ABS function
@ CHECK-RELOCATIONS: ]

