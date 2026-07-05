# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/MachO_global_absolute_def.o %S/Inputs/MachO_global_absolute_def.s
# RUN: llvm-mc -triple x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/MachO_local_absolute_def.o %S/Inputs/MachO_local_absolute_def.s
# RUN: llvm-mc -triple x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/MachO_absolute_symbols.o %s
# RUN: llvm-jitlink -noexec -check=%s %t/MachO_absolute_symbols.o \
# RUN:   %t/MachO_global_absolute_def.o %t/MachO_local_absolute_def.o
#
# Check that both global and local absolute defs work as expected (global
# absolutes visible, local ones not).
#
# jitlink-check: *{4}_GlobalAbsoluteSymDefValue = 0x01234567
# jitlink-check: *{4}_LocalAbsoluteSymDefValue = 0x89ABCDEF

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 10, 14	sdk_version 10, 14
	.globl	_main
	.p2align	4, 0x90
_main:
	retq

	.section	__DATA,__data
# Take the address of GlobalAbsoluteSymDefValue and LocalAbsoluteSymDefValue
# to force linking of the extra input files.
	.globl	extra_files_anchor
	.p2align	3
extra_files_anchor:
        .quad   _GlobalAbsoluteSymDefValue
	.quad	_LocalAbsoluteSymDefValue

.subsections_via_symbols
