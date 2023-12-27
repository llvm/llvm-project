# RUN: llvm-mc -triple arm64-pc-linux-gnu %s -filetype=obj -o /dev/null 2>&1 | FileCheck %s --allow-empty --check-prefix=ELF
# RUN: llvm-mc -triple arm64-windows-gnu %s -filetype=obj -o /dev/null 2>&1 | FileCheck %s --allow-empty --check-prefix=ELF

# REQUIRES: aarch64-registered-target

	.globl	_locomotive
	.p2align	2
_locomotive:
	.cfi_startproc
	ret

	.globl	_caboose
	.p2align	2
_caboose:
	ret
	.cfi_endproc

# Check that the diagnostic does not fire on ELF, nor COFF platforms, which do
# not support subsections_via_symbols. See also: cfi-bad-nesting-darwin.s
# ELF-NOT: error: