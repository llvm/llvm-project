# REQUIRES: asserts
# RUN: llvm-mc -triple=x86_64-apple-darwin11 -filetype=obj -o %t %s
# RUN: llvm-jitlink -noexec -debug-only=jitlink %t 2>&1 | FileCheck %s
#
# Check that splitting of compact-unwind sections works.
#
# CHECK: splitting {{.*}} __LD,__compact_unwind containing 1 initial blocks...
# CHECK:   Splitting {{.*}} into 1 compact unwind record(s)
# CHECK:     Updating {{.*}} to point to _main {{.*}}

	.section	__TEXT,__text,regular,pure_instructions
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

.subsections_via_symbols
