# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/MachO_linker_private_def.o %S/Inputs/MachO_linker_private_def.s
# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/MachO_linker_private_symbols.o %s
# RUN: llvm-jitlink -noexec %t/MachO_linker_private_def.o \
# RUN: %t/MachO_linker_private_symbols.o
#
# Check that we can resolve linker-private symbol definitions across object
# boundaries.

	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 14
	.globl	_main
	.p2align	4, 0x90
_main:
	jmp	l_foo

.subsections_via_symbols
