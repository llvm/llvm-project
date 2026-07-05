@ RUN: not llvm-mc -triple=armv7-unknown-linux-gnueabi < %s 2> %t
@ RUN: FileCheck < %t %s

@ Check the diagnostics for .setfp directive.


	.syntax unified
	.text

@-------------------------------------------------------------------------------
@ TEST1: .setfp before .fnstart
@-------------------------------------------------------------------------------
	.globl	func1
	.align	2
	.type	func1,%function
	.setfp	fp, sp, #0
@ CHECK: error: .fnstart must precede .setfp directive
@ CHECK:        .setfp fp, sp, #0
@ CHECK:        ^
	.fnstart
func1:
	.fnend



@-------------------------------------------------------------------------------
@ TEST2: .setfp after .handlerdata
@-------------------------------------------------------------------------------
	.globl	func2
	.align	2
	.type	func2,%function
	.fnstart
func2:
	.handlerdata
	.setfp	fp, sp, #0
@ CHECK: error: .setfp must precede .handlerdata directive
@ CHECK:        .setfp fp, sp, #0
@ CHECK:        ^
	.fnend



@-------------------------------------------------------------------------------
@ TEST3: .setfp with bad fp register
@-------------------------------------------------------------------------------
	.globl	func3
	.align	2
	.type	func3,%function
	.fnstart
func3:
	.setfp	0, r0, #0
@ CHECK: error: frame pointer register expected
@ CHECK:        .setfp 0, r0, #0
@ CHECK:               ^
	.fnend



@-------------------------------------------------------------------------------
@ TEST4: .setfp with bad sp register
@-------------------------------------------------------------------------------
	.globl	func4
	.align	2
	.type	func4,%function
	.fnstart
func4:
	.setfp	fp, 0, #0
@ CHECK: error: stack pointer register expected
@ CHECK:        .setfp fp, 0, #0
@ CHECK:                   ^
	.fnend



@-------------------------------------------------------------------------------
@ TEST5: .setfp with non-sp register as second operand
@-------------------------------------------------------------------------------
	.globl	func5
	.align	2
	.type	func5,%function
	.fnstart
func5:
	.setfp	fp, r0, #0
@ CHECK: error: register should be either $sp or the latest fp register
@ CHECK:        .setfp fp, r0, #0
@ CHECK:                   ^
	.fnend
