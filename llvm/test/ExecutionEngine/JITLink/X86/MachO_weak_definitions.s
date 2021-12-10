# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/MachO_extra_def_weak.o %S/Inputs/MachO_extra_def_weak.s
# RUN: llvm-mc -triple x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/MachO_weak_definitions.o %s
# RUN: llvm-jitlink -noexec -check=%s %t/MachO_weak_definitions.o \
# RUN:   %t/MachO_extra_def_weak.o
#
# Check that objects linked separately agree on the address of weak symbols.
#
# jitlink-check: *{8}ExtraDefAddrInThisFile = *{8}ExtraDefAddrInExtraFile

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 10, 14	sdk_version 10, 14
	.globl	_main
	.p2align	4, 0x90
_main:
	retq

	.section	__DATA,__data
	.globl	ExtraDef
	.weak_definition	ExtraDef
	.p2align	2
ExtraDef:
	.long	1

	.globl	ExtraDefAddrInThisFile
	.p2align	3
ExtraDefAddrInThisFile:
	.quad	ExtraDef

# Take the address of ExtraDefAddrInExtraFile to force its materialization
	.globl	extra_file_anchor
	.p2align	3
extra_file_anchor:
	.quad	ExtraDefAddrInExtraFile


.subsections_via_symbols
