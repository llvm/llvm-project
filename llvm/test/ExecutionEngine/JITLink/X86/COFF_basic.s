# RUN: llvm-mc -filetype=obj -triple=x86_64-windows-msvc %s -o %t
# RUN: llvm-jitlink -noexec %t
# 
# Check a basic COFF object file loads successfully.

	.text
	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.file	"main.c"
	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main
	.p2align	4, 0x90
main:
    .seh_proc main
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movl	$0, 4(%rsp)
	xorl	%eax, %eax
	popq	%rcx
	retq
	.seh_endproc
