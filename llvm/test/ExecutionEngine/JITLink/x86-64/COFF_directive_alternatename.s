# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t
# RUN: llvm-jitlink -noexec %t
#
# Check object with alternatename directive does not fail, because
# foo external symbol was resolved to foo_def.
#

	.text
	
	.def	 foo_def;
	.scl	2;
	.type	32;
	.endef
	.globl	foo_def
	.p2align	4, 0x90
foo_def:
	retq

	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main
	.p2align	4, 0x90
main:
	callq foo
	retq

.section .drectve,"yn"
.ascii "/alternatename:foo=foo_def"
