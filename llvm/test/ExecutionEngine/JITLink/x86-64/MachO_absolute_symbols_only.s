# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj -o %t/main.o \
# RUN:   %t/main.s
# RUN: llvm-mc -triple=x86_64-apple-macosx10.9 -filetype=obj -o %t/abs.o \
# RUN:   %t/abs.s
# RUN: llvm-jitlink -noexec -check=%t/main.s %t/main.o %t/abs.o
#
# Check that an object whose only definitions are absolute symbols links.
# Its graph has an empty __text section and needs no memory.

#--- main.s
# jitlink-check: *{8}ptr = 0x53

	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main
_main:
	retq

	.section	__DATA,__data
	.globl	ptr
	.p2align	3
ptr:
	.quad	_abs

#--- abs.s
	.globl	_abs
_abs = 0x53
