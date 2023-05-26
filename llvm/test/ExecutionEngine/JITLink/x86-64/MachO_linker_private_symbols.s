# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/global_lp_def.o %S/Inputs/MachO_global_linker_private_def.s
# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/internal_lp_def.o %S/Inputs/MachO_internal_linker_private_def.s
# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/macho_lp_test.o %s
# RUN: llvm-jitlink -noexec %t/global_lp_def.o %t/macho_lp_test.o
# RUN: not llvm-jitlink -noexec %t/internal_lp_def.o %t/macho_lp_test.o
#
# Check that we can resolve global symbols whose names start with the
# linker-private prefix 'l' across object boundaries, and that we can't resolve
# internals with the linker-private prefix across object boundaries.

	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 14
	.globl	_main
	.p2align	4, 0x90
_main:
	jmp	l_foo

.subsections_via_symbols
