@ RUN: llvm-mc -triple thumbv7-windows-itanium -filetype obj -o - %s \
@ RUN:   | llvm-readobj -S - | FileCheck %s

	.syntax unified
	.thumb

	.text

	.def function
		.type 32
		.scl 2
	.endef
	.global function
	.thumb_func
function:
	bx lr

@ CHECK: Sections [
@ CHECK:   Section {
@ CHECK:     Name: .text
@ CHECK:     Characteristics [
@ CHECK:       IMAGE_SCN_ALIGN_4BYTES
@ CHECK:       IMAGE_SCN_CNT_CODE
@ CHECK:       IMAGE_SCN_MEM_16BIT
@ CHECK:       IMAGE_SCN_MEM_EXECUTE
@ CHECK:       IMAGE_SCN_MEM_PURGEABLE
@ CHECK:       IMAGE_SCN_MEM_READ
@ CHECK:     ]
@ CHECK:   }
@ CHECK: ]
