@ RUN: not llvm-mc %s -triple=armv7-unknown-linux-gnueabi \
@ RUN:   -filetype=obj -o /dev/null 2>&1 | FileCheck %s

@ Check the diagnostics for mismatched .fnend directive


	.syntax unified
	.text

	.globl	func1
	.align	2
	.type	func1,%function
func1:
	.fnend
@ CHECK: error: .fnstart must precede .fnend directive
@ CHECK:        .fnend
@ CHECK:        ^
