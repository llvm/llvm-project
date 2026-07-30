@ RUN: not llvm-mc -triple=armv7-unknown-linux-gnueabi < %s 2> %t
@ RUN: FileCheck < %t %s

@ Check the diagnostics for .vsave directive

@ .vsave directive should always come after .fnstart directive
@ and before .handlerdata directive.

	.syntax unified
	.text

@-------------------------------------------------------------------------------
@ TEST1: .vsave before .fnstart
@-------------------------------------------------------------------------------
	.globl	func1
	.align	2
	.type	func1,%function
	.vsave	{d0, d1, d2, d3}
@ CHECK: error: .fnstart must precede .save or .vsave directives
@ CHECK:        .vsave {d0, d1, d2, d3}
@ CHECK:        ^
	.fnstart
func1:
	.fnend



@-------------------------------------------------------------------------------
@ TEST2: .vsave after .handlerdata
@-------------------------------------------------------------------------------
	.globl	func2
	.align	2
	.type	func2,%function
	.fnstart
func2:
	.handlerdata
	.vsave	{d0, d1, d2, d3}
@ CHECK: error: .save or .vsave must precede .handlerdata directive
@ CHECK:        .vsave {d0, d1, d2, d3}
@ CHECK:        ^
	.fnend
