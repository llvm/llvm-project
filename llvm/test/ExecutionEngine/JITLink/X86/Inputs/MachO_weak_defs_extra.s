# Supplies a weak def, WeakDef, and a pointer holding its address,
# WeakDefAddrInExtraFile.

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 10, 14	sdk_version 10, 14
	.section	__DATA,__data
	.globl	WeakDef
	.weak_definition	WeakDef
	.p2align	2
WeakDef:
	.long	2

	.globl	WeakDefAddrInExtraFile
	.p2align	3
WeakDefAddrInExtraFile:
	.quad	WeakDef


.subsections_via_symbols
