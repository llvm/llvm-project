# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj -o %t %s
# RUN: llvm-jitlink -noexec %t
#
# Check that JITLink handles anonymous relocations to the end of MachO sections.

	.section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 11, 0	sdk_version 11, 1
	.globl	_main
	.p2align	4, 0x90
_main:

	movq	_R(%rip), %rax
	retq

	.section	__TEXT,__anon,regular
L__anon_start:
        .byte 7
L__anon_end:

	.private_extern	_R
	.section	__DATA,__data
	.globl	_R
	.p2align	3
_R:
	.quad L__anon_end

.subsections_via_symbols
