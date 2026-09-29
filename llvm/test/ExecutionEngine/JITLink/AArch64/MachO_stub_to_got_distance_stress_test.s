# RUN: llvm-mc -triple=arm64-apple-darwin19 -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -phony-externals %t.o
#
# Use RW- empty space sufficient to push the R-- and R-X segments more than
# 2^20 bytes apart. This will cause the LDRLiteral19 relocations from the STUB
# section to the GOT to overflow if not handled correctly.

        .section	__TEXT,__text,regular,pure_instructions
	.ios_version_min 7, 0	sdk_version 16, 0
	.globl	_main
	.p2align	2
_main:
	b	_foo

	.comm	_empty_space,2097152,0

.subsections_via_symbols
