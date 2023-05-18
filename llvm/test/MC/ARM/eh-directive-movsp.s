@ RUN: llvm-mc -triple armv7-eabi -filetype obj -o - %s \
@ RUN:   | llvm-readobj -S --sd - \
@ RUN:   | FileCheck %s

	.syntax unified
	.thumb

	.section .duplicate

	.global duplicate
	.type duplicate,%function
duplicate:
	.fnstart
	.setfp sp, sp, #8
	add sp, sp, #8
	.movsp r11
	mov r11, sp
	.fnend

@ CHECK: Section {
@ CHECK:   Name: .ARM.exidx.duplicate
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 B09B9B80
@ CHECK:   )
@ CHECK: }


	.section .squash

	.global squash
	.type squash,%function
squash:
	.fnstart
	.movsp ip
	mov ip, sp
	.save {fp, ip, lr}
	stmfd sp!, {fp, ip, lr}
	.fnend

@ CHECK: Section {
@ CHECK:   Name: .ARM.exidx.squash
@ CHECK:   SectionData (
@ CHECK:     0000: 00000000 9C808580
@ CHECK:   )
@ CHECK: }
