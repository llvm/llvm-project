@ RUN: llvm-mc -triple armv7-eabi -mattr hwdiv -filetype asm -o /dev/null %s

	.syntax unified
	.thumb

	udiv r0, r1, r2
	.arch_extension idiv
	udiv r0, r1, r2
