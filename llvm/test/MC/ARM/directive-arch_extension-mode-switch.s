@ RUN: not llvm-mc -triple armv8-eabi -filetype asm -o /dev/null %s 2>&1 | FileCheck %s

@ Ensure that a mode switch does not revert the architectural features that were
@ alternated explicitly.

	.syntax unified

	.arch_extension noidiv

	.arm
	udiv r0, r0, r1
@ CHECK: instruction requires: divide in ARM

	.thumb
	udiv r0, r0, r1
@ CHECK: instruction requires: divide in THUMB

