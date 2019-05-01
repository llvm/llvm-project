// RUN: llvm-mc < %s -filetype=obj -triple x86_64-apple-macosx10.8.0 | llvm-readobj -S | FileCheck -check-prefix=MACHO %s
// RUN: llvm-mc < %s -filetype=obj -triple x86_64-apple-ios7.0.0 | llvm-readobj -S | FileCheck -check-prefix=MACHO %s
// RUN: llvm-mc < %s -filetype=obj -triple x86_64-unknown-linux | llvm-readobj -S | FileCheck -check-prefix=ELF %s

	.globl	__Z3barv
	.align	4, 0x90
__Z3barv:
	.cfi_startproc
	pushq	%rax
	.cfi_def_cfa_offset 16
	popq	%rax
	retq
	.cfi_endproc

// MACHO: Name: __compact_unwind
// ELF-NOT: __compact_unwind
