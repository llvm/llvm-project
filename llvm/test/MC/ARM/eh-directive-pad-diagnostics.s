@ RUN: not llvm-mc -triple=armv7-unknown-linux-gnueabi < %s 2> %t
@ RUN: FileCheck < %t %s

@ Check the diagnostics for .pad directive.


	.syntax unified
	.text

@-------------------------------------------------------------------------------
@ TEST1: .pad before .fnstart
@-------------------------------------------------------------------------------
	.globl	func1
	.align	2
	.type	func1,%function
	.pad	#0
@ CHECK: error: .fnstart must precede .pad directive
@ CHECK:        .pad #0
@ CHECK:        ^
	.fnstart
func1:
	.fnend



@-------------------------------------------------------------------------------
@ TEST2: .pad after .handlerdata
@-------------------------------------------------------------------------------
	.globl	func2
	.align	2
	.type	func2,%function
	.fnstart
func2:
	.handlerdata
	.pad	#0
@ CHECK: error: .pad must precede .handlerdata directive
@ CHECK:        .pad #0
@ CHECK:        ^
	.fnend
