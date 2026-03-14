@ RUN: not llvm-mc -triple thumbv7-eabi -mattr +thumb2 %s 2>&1 | FileCheck %s

	.syntax unified
	.text
	.thumb

undefined:
	udfpl

@ CHECK: error: instruction 'udf' is not predicable, but condition code specified
@ CHECK: 	udfpl
@ CHECK: 	^

	udf #256

@ CHECK: error: invalid instruction, any one of the following would fix this:
@ CHECK: note: instruction requires: arm-mode
@ CHECK: note: operand must be an immediate in the range [0,255]
@ CHECK: 	udf #256
@ CHECK: 	^

	udf.w #65536

@ CHECK: error: operand must be an immediate in the range [0,65535]
@ CHECK: 	udf.w #65536
@ CHECK: 	      ^

