@ RUN: llvm-mc -triple armv7-eabi -filetype asm -o - %s | FileCheck %s

	.syntax unified
	.fpu vfp

	.type aliases,%function
aliases:
	fstmeax sp!, {d0}
	fldmfdx sp!, {d0}

	fstmfdx sp!, {d0}
	fldmeax sp!, {d0}

@ CHECK-LABEL: aliases
@ CHECK: 	fstmiax sp!, {d0}
@ CHECK: 	fldmiax sp!, {d0}
@ CHECK: 	fstmdbx sp!, {d0}
@ CHECK: 	fldmdbx sp!, {d0}

	fstmiaxcs r0, {d0}
	fstmiaxhs r0, {d0}
	fstmiaxls r0, {d0}
	fstmiaxvs r0, {d0}
@ CHECK: 	fstmiaxhs r0, {d0}
@ CHECK: 	fstmiaxhs r0, {d0}
@ CHECK: 	fstmiaxls r0, {d0}
@ CHECK: 	fstmiaxvs r0, {d0}

