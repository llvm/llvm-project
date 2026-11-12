# Supplies a strong definition of ExtraDef.

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 10, 14	sdk_version 10, 14
	.section	__DATA,__data
	.globl	ExtraDef
	.p2align	2
ExtraDef:
	.long	3

.subsections_via_symbols
