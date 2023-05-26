# REQUIRES: system-darwin && asserts
# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/MachO_extra_def_strong.o %S/Inputs/MachO_extra_def_strong.s
# RUN: llvm-mc -triple x86_64-apple-macosx10.9 -filetype=obj \
# RUN:   -o %t/MachO_gdb_jit_nonzero_alignment_offsets.o %s
# RUN: llvm-jitlink -noexec -debugger-support \
# RUN:   %t/MachO_gdb_jit_nonzero_alignment_offsets.o \
# RUN:   %t/MachO_extra_def_strong.o
#
# Check that blocks with non-zero alignment offsets don't break debugging
# support.
#
# In this test case the ExtraDef symbol below will be overridden by a strong
# def in MachO_strong_defs_extra.s. This will leave main (with alignment 16,
# alignment offset 4) as the only block in __TEXT,__text. The testcase ensures
# that the debugging support plugin doesn't crash or throw an error in this
# case.

        .section	__TEXT,__text,regular,pure_instructions
	.p2align	4, 0x90

	.globl	ExtraDef
	.weak_definition	ExtraDef
	.p2align	2
ExtraDef:
	.long	42

	.globl	_main
_main:
	xorq	%rax, %rax
	retq

	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.asciz	"test dwarf string"

.subsections_via_symbols
