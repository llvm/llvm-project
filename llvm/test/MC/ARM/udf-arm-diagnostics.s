@ RUN: not llvm-mc -triple arm-eabi %s 2>&1 | FileCheck %s

	.syntax unified
	.text
	.arm

undefined:
	udfpl

@ CHECK: error: instruction 'udf' is not predicable, but condition code specified
@ CHECK: 	udfpl
@ CHECK: 	^

	udf #65536

@ CHECK: error: operand must be an immediate in the range [0,65535]
@ CHECK: 	udf #65536
@ CHECK: 	    ^

