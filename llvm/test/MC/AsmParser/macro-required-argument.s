# RUN: llvm-mc -triple i386 -o - %s | FileCheck %s

	.macro required parameter:req
		.long \parameter
	.endm

	required 0
# CHECK: .long 0

	.macro required_with_default parameter:req=0
		.long \parameter
	.endm

	required 1
# CHECK: .long 1

