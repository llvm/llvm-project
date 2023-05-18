@ RUN: not llvm-mc -triple=armv7-unknown-linux-gnueabi < %s 2> %t
@ RUN: FileCheck < %t %s

@ Check the diagnostics for .personality directive.


	.syntax unified
	.text

@-------------------------------------------------------------------------------
@ TEST1: .personality before .fnstart
@-------------------------------------------------------------------------------
	.globl	func1
	.align	2
	.type	func1,%function
	.personality	__gxx_personality_v0
@ CHECK: error: .fnstart must precede .personality directive
@ CHECK:        .personality __gxx_personality_v0
@ CHECK:        ^
	.fnstart
func1:
	.fnend



@-------------------------------------------------------------------------------
@ TEST2: .personality after .handlerdata
@-------------------------------------------------------------------------------
	.globl	func2
	.align	2
	.type	func2,%function
	.fnstart
func2:
	.handlerdata
	.personality	__gxx_personality_v0
@ CHECK: error: .personality must precede .handlerdata directive
@ CHECK:        .personality __gxx_personality_v0
@ CHECK:        ^
	.fnend
