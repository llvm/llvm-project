@ RUN: llvm-mc < %s -triple thumbv5-linux-gnueabi -filetype=obj -o - \
@ RUN:   | llvm-readobj -r - | FileCheck %s
	.syntax	unified

	.text
	.align	2
	.globl	main
	.type	main,%function
	.thumb_func
main:
	bl	end
	.space 8192
end:
	bl	main2
	bx	lr

	.text
	.align	2
	.globl	main2
	.type	main2,%function
	.thumb_func
main2:
	bx	lr

@ CHECK-NOT: 0x0 R_ARM_THM_CALL end
@ CHECK: 0x2004 R_ARM_THM_CALL main2
