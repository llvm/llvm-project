@ RUN: llvm-mc < %s -triple armv7-unknown-linux-gnueabi -filetype=obj -o /dev/null

@ Check softvfp as the FPU name.

@ Expected result: The integrated-as should be able to assemble this file
@ without problems.

	.fpu	softvfp
