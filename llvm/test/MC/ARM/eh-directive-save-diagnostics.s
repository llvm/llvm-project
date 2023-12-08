@ RUN: not llvm-mc -triple=armv7-unknown-linux-gnueabi < %s 2> %t
@ RUN: FileCheck < %t %s

@ Check the diagnostics for .save directive

@ .save directive should always come after .fnstart directive and
@ before .handlerdata directive.

	.syntax unified
	.text

@-------------------------------------------------------------------------------
@ TEST1: .save before .fnstart
@-------------------------------------------------------------------------------
	.globl	func1
	.align	2
	.type	func1,%function
	.save	{r4, r5, r6, r7}
@ CHECK: error: .fnstart must precede .save or .vsave directives
@ CHECK:        .save {r4, r5, r6, r7}
@ CHECK:        ^
	.fnstart
func1:
	.fnend



@-------------------------------------------------------------------------------
@ TEST2: .save after .handlerdata
@-------------------------------------------------------------------------------
	.globl	func2
	.align	2
	.type	func2,%function
	.fnstart
func2:
	.handlerdata
	.save	{r4, r5, r6, r7}
@ CHECK: error: .save or .vsave must precede .handlerdata directive
@ CHECK:        .save {r4, r5, r6, r7}
@ CHECK:        ^
	.fnend
