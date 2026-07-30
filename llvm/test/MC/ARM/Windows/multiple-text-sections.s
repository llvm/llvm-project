@ RUN: llvm-mc -triple thumbv7-windows-itanium -filetype obj -o - %s \
@ RUN:   | llvm-readobj -S - | FileCheck %s

	.syntax unified
	.text
	.thumb

	.section	.text,"xr",one_only,a

	.def	 a;
		.scl	2;
		.type	32;
	.endef
a:
	movs	r0, #65
	bx	lr

	.section	.text,"xr",one_only,b

	.def	 b;
		.scl	2;
		.type	32;
	.endef
	.thumb_func
b:
	movs	r0, #66
	bx	lr

@ CHECK: Sections [
@ CHECK:   Section {
@ CHECK:     Name: .text
@ CHECK:     Characteristics [
@ CHECK:       IMAGE_SCN_CNT_CODE
@ CHECK:       IMAGE_SCN_MEM_16BIT
@ CHECK:       IMAGE_SCN_MEM_EXECUTE
@ CHECK:       IMAGE_SCN_MEM_READ
@ CHECK:     ]
@ CHECK:   }
@ CHECK:   Section {
@ CHECK:     Name: .text
@ CHECK:     Characteristics [
@ CHECK:       IMAGE_SCN_CNT_CODE
@ CHECK:       IMAGE_SCN_MEM_16BIT
@ CHECK:       IMAGE_SCN_MEM_EXECUTE
@ CHECK:       IMAGE_SCN_MEM_READ
@ CHECK:     ]
@ CHECK:   }
@ CHECK:   Section {
@ CHECK:     Name: .text
@ CHECK:     Characteristics [
@ CHECK:       IMAGE_SCN_CNT_CODE
@ CHECK:       IMAGE_SCN_MEM_16BIT
@ CHECK:       IMAGE_SCN_MEM_EXECUTE
@ CHECK:       IMAGE_SCN_MEM_READ
@ CHECK:     ]
@ CHECK:   }
@ CHECK: ]

