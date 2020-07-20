@ Check the value of st_other for thumb function.

@ ARM does not define any st_other flags for thumb function.  The value
@ for st_other should always be 0.

@ RUN: llvm-mc < %s -triple thumbv5-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj --symbols - | FileCheck %s

	.syntax	unified
	.text
	.align	2
	.thumb_func
	.global	main
	.type	main,%function
main:
	bx	lr

@ CHECK: Name: main
@ CHECK: Other: 0
