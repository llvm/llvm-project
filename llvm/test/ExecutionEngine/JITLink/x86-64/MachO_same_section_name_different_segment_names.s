# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj -o %t %s
# RUN: llvm-jitlink -noexec %t
#
# Check that JITLink handles MachO sections with the same section name but
# different segment names.

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 11, 0	sdk_version 11, 1
	.globl	_main
	.p2align	4, 0x90
_main:                                  ## @main
	xorl	%eax, %eax
	retq

        .section	__TEXT,__const
        .globl _a
_a:
        .quad   42

	.section	__DATA,__const
	.globl	_b
	.p2align	3
_b:
	.quad	42

.subsections_via_symbols
