# REQUIRES: asserts
# RUN: llvm-mc -triple=x86_64-apple-macos10.9 -filetype=obj -o %t %s
# RUN: llvm-jitlink -noexec %t
#
# Verify that PC-begin candidate symbols have been sorted correctly when adding
# PC-begin edges for FDEs. In this test both _main and _X are at address zero,
# however we expect to select _main over _X as _X is common. If the sorting
# fails we'll trigger an assert in EHFrameEdgeFixer, otherwise this test will
# succeed.

        .section	__TEXT,__text,regular,pure_instructions
	.build_version macos, 12, 0	sdk_version 13, 0
	.globl	_main
	.p2align	4, 0x90
_main:
	.cfi_startproc
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	xorl	%eax, %eax
	popq	%rbp
	retq
	.cfi_endproc

	.comm	_X,4,2
.subsections_via_symbols
