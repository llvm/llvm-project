@ RUN: not llvm-mc -triple armv7-eabi -filetype asm -o /dev/null %s 2>&1 \
@ RUN:   | FileCheck %s

	.syntax unified
	.fpu vfp

	.type aliases,%function
aliases:
	fstmeax sp!, {s0}
	fldmfdx sp!, {s0}

	fstmfdx sp!, {s0}
	fldmeax sp!, {s0}

@ CHECK-LABEL: aliases
@ CHECK: error: operand must be a list of registers in range [d0, d31]
@ CHECK:	fstmeax sp!, {s0}
@ CHECK:                     ^
@ CHECK: error: operand must be a list of registers in range [d0, d31]
@ CHECK:	fldmfdx sp!, {s0}
@ CHECK:                     ^

@ CHECK: error: operand must be a list of registers in range [d0, d31]
@ CHECK:	fstmfdx sp!, {s0}
@ CHECK:                     ^
@ CHECK: error: operand must be a list of registers in range [d0, d31]
@ CHECK:	fldmeax sp!, {s0}
@ CHECK:                     ^

	fstmiaxcs r0, {s0}
	fstmiaxhs r0, {s0}
	fstmiaxls r0, {s0}
	fstmiaxvs r0, {s0}
@ CHECK: error: operand must be a list of registers in range [d0, d31]
@ CHECK: 	fstmiaxcs r0, {s0}
@ CHECK:                      ^
@ CHECK: error: operand must be a list of registers in range [d0, d31]
@ CHECK: 	fstmiaxhs r0, {s0}
@ CHECK:                      ^
@ CHECK: error: operand must be a list of registers in range [d0, d31]
@ CHECK: 	fstmiaxls r0, {s0}
@ CHECK:                      ^
@ CHECK: error: operand must be a list of registers in range [d0, d31]
@ CHECK: 	fstmiaxvs r0, {s0}
@ CHECK:                      ^

