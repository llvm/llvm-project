# REQUIRES: asserts
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t
# RUN: llvm-jitlink -num-threads=0 -debug-only=jitlink -noexec \
# RUN:              -abs func=0xcafef00d %t 2>&1 \
# RUN:              | FileCheck %s
#
# Check a file debug symbol is skipped.
#
# CHECK: Creating graph symbols...
# CHECK:   7: Skipping FileRecord symbol ".file" in (debug) (index: -2)

	.text

	.file	"skip_this_file_symbol"

	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main
	.p2align	4, 0x90
main:
	retq
