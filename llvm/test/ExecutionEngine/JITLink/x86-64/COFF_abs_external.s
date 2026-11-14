# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc -o %t/main.o %t/main.s
# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc -o %t/abs.o %t/abs.s
# RUN: llvm-jitlink -noexec -check=%t/main.s %t/main.o %t/abs.o
#
# Check that an external absolute symbol is visible to other objects.

#--- main.s
# jitlink-check: *{8}ptr = 0x53

	.text
	.globl	main
main:
	retq

	.data
	.globl	ptr
	.p2align	3
ptr:
	.quad	abs

#--- abs.s
	.globl	abs
abs = 0x53
