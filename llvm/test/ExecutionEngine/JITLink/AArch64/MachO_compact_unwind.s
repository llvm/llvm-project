# REQUIRES: asserts
# RUN: llvm-mc -triple=arm64-apple-ios -filetype=obj -o %t %s
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec %t 2>&1 \
# RUN:              | FileCheck %s
#
# Check that splitting of compact-unwind sections works.
#
# CHECK: splitting {{.*}} __LD,__compact_unwind containing 1 initial blocks...
# CHECK:   Splitting {{.*}} into 1 compact unwind record(s)
# CHECK:     Updating {{.*}} to point to _main {{.*}}

	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
	.p2align	2
_main:
	.cfi_startproc
	ret
	.cfi_endproc

.subsections_via_symbols

