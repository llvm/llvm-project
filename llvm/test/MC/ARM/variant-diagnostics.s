@ RUN: not llvm-mc -triple armv7-linux-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s

	.arch armv7

	.type invalid_variant,%function
invalid_variant:
	bx target(invalid)

@ CHECK: error: invalid variant 'invalid'
@ CHECK: 	bx target(invalid)
@ CHECK:                  ^

