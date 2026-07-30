	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 14
	.globl	bypass_got
	.p2align	4, 0x90
bypass_got:
	movq	_x@GOTPCREL(%rip), %rax

.subsections_via_symbols
