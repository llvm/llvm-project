# Supplies a strong definition of WeakDef.

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 10, 14	sdk_version 10, 14
	.section	__DATA,__data
	.globl	WeakDef
	.p2align	2
WeakDef:
	.long	3

.subsections_via_symbols
