# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/MachO_weak_defs_extra.o %S/Inputs/MachO_weak_defs_extra.s
# RUN: llvm-mc -triple x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/MachO_weak_definitions.o %s
# RUN: llvm-jitlink -noexec -check=%s %t/MachO_weak_definitions.o \
# RUN:   %t/MachO_weak_defs_extra.o
#
# Check that objects linked separately agree on the address of weak symbols.
#
# jitlink-check: *{8}WeakDefAddrInThisFile = *{8}WeakDefAddrInExtraFile

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 10, 14	sdk_version 10, 14
	.globl	_main
	.p2align	4, 0x90
_main:
	retq

	.section	__DATA,__data
	.globl	WeakDef
	.weak_definition	WeakDef
	.p2align	2
WeakDef:
	.long	1

	.globl	WeakDefAddrInThisFile
	.p2align	3
WeakDefAddrInThisFile:
	.quad	WeakDef

# Take the address of WeakDefAddrInExtraFile to force its materialization
	.globl	extra_file_anchor
	.p2align	3
extra_file_anchor:
	.quad	WeakDefAddrInExtraFile


.subsections_via_symbols
