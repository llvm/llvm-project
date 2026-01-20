# RUN: llvm-mc -triple=x86_64-unknown-freebsd -position-independent \
# RUN:     -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -show-graphs=".*" %t.o | FileCheck %s

# Make sure that the LinkGraph triple correctly reports FreeBSD (tests
# ba6a59c8750).

# CHECK: LinkGraph {{.*}} (triple = x86_64{{.*}}freebsd)

	.text
	.globl	main
	.p2align	4
	.type	main,@function
main:
	xorl	%eax, %eax
	retq
