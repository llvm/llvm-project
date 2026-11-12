@ RUN: llvm-mc -triple armv7-eabi -filetype asm -o - %s | FileCheck %s

	.syntax unified

	.data

	.type .L_table_begin,%object
.L_table_begin:
	.rep 2
	.long 0xd15ab1ed
	.long 0x0ff1c1a1
	.endr
.L_table_end:

	.text

	.type return,%function
return:
	bx lr

	.global arm_function
	.type arm_function,%function
arm_function:
	mov r0, #:lower16:((.L_table_end - .L_table_begin) >> 2)
	blx return

@ CHECK-LABEL: arm_function
@ CHECK:  	movw r0, :lower16:((.L_table_end-.L_table_begin)>>2)
@ CHECK:  	blx return

	.global thumb_function
	.type thumb_function,%function
thumb_function:
	mov r0, #:lower16:((.L_table_end - .L_table_begin) >> 2)
	blx return

@ CHECK-LABEL: thumb_function
@ CHECK:  	movw r0, :lower16:((.L_table_end-.L_table_begin)>>2)
@ CHECK:  	blx return

