# Supplies a weak def of ExtraDef, and a pointer holding its address,
# ExtraDefInExtraFile.

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 10, 14	sdk_version 10, 14
	.section	__DATA,__data
	.globl	ExtraDef
	.weak_definition	ExtraDef
	.p2align	2
ExtraDef:
	.long	2

	.globl	ExtraDefAddrInExtraFile
	.p2align	3
ExtraDefAddrInExtraFile:
	.quad	ExtraDef


.subsections_via_symbols
