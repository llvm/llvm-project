# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t
# RUN: llvm-jitlink -noexec -check=%s %t
#
# Check that a weak external with a data default links and resolves to its
# default, and that a weak alias of a weak function links.
#
# jitlink-check: *{4}wdata = 42
# jitlink-check: a = b

	.text
	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main
	.p2align	4, 0x90
main:
	retq

	.def	b;
	.scl	2;
	.type	32;
	.endef
	.weak	b
b:
	retq

	.weak	a
	.def	a;
	.scl	2;
	.type	32;
	.endef
	.set	a, b

	.data
	.weak	wdata
	.p2align	2
wdata:
	.long	42
