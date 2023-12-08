# Supplies a weak def, WeakDef, and a pointer holding its address,
# WeakDefAddrInExtraFile.

	.text
	.file	"ELF_weak_defs_extra.c"
	.type	WeakDef,@object
	.data
	.weak	WeakDef
	.p2align	2
WeakDef:
	.long	2
	.size	WeakDef, 4

	.type	WeakDefAddrInExtraFile,@object
	.globl	WeakDefAddrInExtraFile
	.p2align	3
WeakDefAddrInExtraFile:
	.quad	WeakDef
	.size	WeakDefAddrInExtraFile, 8

	.ident	"clang version 10.0.0-4ubuntu1 "
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym WeakDef
