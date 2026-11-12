@ RUN: llvm-mc -triple arm-unknown-linux -filetype=obj %s -o %t
@ RUN: llvm-objdump -d %t | FileCheck %s
@ RUN: llvm-objdump -d -r %t | FileCheck --check-prefixes=CHECK,RELOC %s

	.cpu arm7tdmi
	.global	myInt
	.data
	.align	2
	.type	myInt, %object
	.size	myInt, 4
myInt:
	.word	1
	.text
	.align	2
	.global	main
	.type	main, %function
main:
	str	fp, [sp, #-4]!
	add	fp, sp, #0
	ldr	r3, .L3
	ldr	r3, [r3]
	mov	r0, r3
	sub	sp, fp, #0
	ldr	fp, [sp], #4
	bx	lr
.L4:
	.align	2
.L3:
	.word	myInt
	.size	main, .-main
        .global myStr
        .type myStr, %object
myStr:
        .string "test string"


@ CHECK:       .word   0x00000000
@ RELOC-NEXT:    R_ARM_ABS32  myInt
@ CHECK-EMPTY:
@ CHECK-NEXT:  <myStr>:
@ CHECK-NEXT:    74 65 73 74 20 73 74 72         test str
