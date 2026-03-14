@ RUN: not llvm-mc -triple armv7-eabi -filetype asm -o /dev/null 2>&1 %s \
@ RUN:   | FileCheck %s

	.syntax unified
	.thumb

	.text

	.global multiple_personality_disorder
	.type multiple_personality_disorder,%function
multiple_personality_disorder:
	.fnstart
	.personality __gcc_personality_v0
	.personality __gxx_personality_v0
	.personality __gxx_personality_sj0
	.cantunwind

@ CHECK: error: .cantunwind can't be used with .personality directive
@ CHECK: .cantunwind
@ CHECK: ^
@ CHECK: note: .personality was specified here
@ CHECK: .personality __gcc_personality_v0
@ CHECK: ^
@ CHECK: note: .personality was specified here
@ CHECK: .personality __gxx_personality_v0
@ CHECK: ^
@ CHECK: note: .personality was specified here
@ CHECK: .personality __gxx_personality_sj0
@ CHECK: ^

