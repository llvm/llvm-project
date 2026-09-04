/// Test the callgraph section to make sure the indirect callsites
/// (annotated by generated temporary labels .Ltmp*) are associated
/// with the corresponding callee type identifiers.

// RUN: llvm-mc -triple=x86_64 -filetype=obj -o - < %s | llvm-readelf -x .llvm.callgraph - | FileCheck %s
	
	.text
	.globl	ball                            # -- Begin function ball
	.p2align	4
	.type	ball,@function
ball:                                   # @ball
.Lfunc_begin0:	
# %bb.0:                                # %entry
	pushq	%rbx	
	subq	$32, %rsp	
	movl	$0, 4(%rsp)
	movq	foo@GOTPCREL(%rip), %rcx
	movq	%rcx, 24(%rsp)
	xorl	%eax, %eax
	callq	*%rcx
.Ltmp0:
	movq	bar@GOTPCREL(%rip), %rax
	movq	%rax, 16(%rsp)
	movsbl	3(%rsp), %edi
	callq	*%rax
.Ltmp1:
	movq	baz@GOTPCREL(%rip), %rax
	movq	%rax, 8(%rsp)
	leaq	3(%rsp), %rbx
	movq	%rbx, %rdi
	callq	*%rax
.Ltmp2:
	callq	foo@PLT
	movsbl	3(%rsp), %edi
	callq	bar@PLT
	movq	%rbx, %rdi
	callq	baz@PLT
	addq	$32, %rsp	
	popq	%rbx	
	retq
	.section	.llvm.callgraph,"o",@progbits,.text
	.quad	0
	.quad	.Lfunc_begin0
	.quad	1
	.quad	3
	/// MD5 hash of the callee type ID for foo.
	// CHECK: 2444f731 f5eecb3e
	.quad	0x3ecbeef531f74424
	.quad	.Ltmp0
	/// MD5 hash of the callee type ID for bar.
	// CHECK: 5486bc59 814b8e30
	.quad	0x308e4b8159bc8654
	.quad	.Ltmp1
	/// MD5 hash of the callee type ID for baz.
	// CHECK: 7ade6814 f897fd77
	.quad	0x77fd97f81468de7a
	.quad	.Ltmp2
	.text
