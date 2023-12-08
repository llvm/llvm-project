@ RUN: not llvm-mc %s -triple=armv7-unknown-linux-gnueabi \
@ RUN:   -filetype=obj -o /dev/null 2>&1 | FileCheck %s

@ Check the diagnostics for the mismatched .fnstart directives.

@ There should be some diagnostics when the previous .fnstart is not closed
@ by the .fnend directive.


	.syntax unified
	.text

	.globl	func1
	.align	2
	.type	func1,%function
	.fnstart
func1:
	@ Intentionally miss the .fnend directive

	.globl	func2
	.align	2
	.type	func2,%function
	.fnstart
@ CHECK: error: .fnstart starts before the end of previous one
@ CHECK:        .fnstart
@ CHECK:        ^
@ CHECK: note: .fnstart was specified here
@ CHECK:        .fnstart
@ CHECK:        ^
func2:
	.fnend
