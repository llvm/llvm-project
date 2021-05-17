# REQUIRES: asserts
# UNSUPPORTED: system-windows
# RUN: llvm-mc -triple=x86_64-unknown-linux -position-independent \
# RUN:     -filetype=obj -o %t %s
# RUN: llvm-jitlink -debug-only=jitlink -define-abs bar=0x01 \
# RUN:     -define-abs _ZTIi=0x02 -noexec %t 2>&1 | FileCheck %s
#
# FIXME: This test should run on windows. Investigate spurious
# 'note: command had no output on stdout or stderr' errors, then re-enable.
#
# Check that a basic .eh-frame section is recognized and parsed. We
# Expect to see two FDEs with corresponding keep-alive edges.
#
# CHECK: Adding keep-alive edge from target at {{.*}} to FDE at
# CHECK: Adding keep-alive edge from target at {{.*}} to FDE at

	.text
	.file	"exceptions.cpp"
	.globl	foo
	.p2align	4, 0x90
	.type	foo,@function
foo:
	.cfi_startproc

	pushq	%rax
	.cfi_def_cfa_offset 16
	movl	$4, %edi
	callq	__cxa_allocate_exception@PLT
	movl	$1, (%rax)
	movq	_ZTIi@GOTPCREL(%rip), %rsi
	movq	%rax, %rdi
	xorl	%edx, %edx
	callq	__cxa_throw@PLT
.Lfunc_end0:
	.size	foo, .Lfunc_end0-foo
	.cfi_endproc

	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 155, DW.ref.__gxx_personality_v0
	.cfi_lsda 27, .Lexception0

	pushq	%rbx
	.cfi_def_cfa_offset 16
	.cfi_offset %rbx, -16
	xorl	%ebx, %ebx
.Ltmp0:
	callq	bar@PLT
.Ltmp1:

	movl	%ebx, %eax
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.LBB1_1:
	.cfi_def_cfa_offset 16
.Ltmp2:
	movq	%rax, %rdi
	callq	__cxa_begin_catch@PLT
	callq	__cxa_end_catch@PLT
	movl	$1, %ebx
	movl	%ebx, %eax
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2
GCC_except_table1:
.Lexception0:
	.byte	255
	.byte	156
	.uleb128 .Lttbase0-.Lttbaseref0
.Lttbaseref0:
	.byte	1
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.uleb128 .Ltmp0-.Lfunc_begin0
	.uleb128 .Ltmp1-.Ltmp0
	.uleb128 .Ltmp2-.Lfunc_begin0
	.byte	1
	.uleb128 .Ltmp1-.Lfunc_begin0
	.uleb128 .Lfunc_end1-.Ltmp1
	.byte	0
	.byte	0
.Lcst_end0:
	.byte	1

	.byte	0
	.p2align	2

.Ltmp3:
	.quad	.L_ZTIi.DW.stub-.Ltmp3
.Lttbase0:
	.p2align	2

	.data
	.p2align	3
.L_ZTIi.DW.stub:
	.quad	_ZTIi
	.hidden	DW.ref.__gxx_personality_v0
	.weak	DW.ref.__gxx_personality_v0
	.section	.data.DW.ref.__gxx_personality_v0,"aGw",@progbits,DW.ref.__gxx_personality_v0,comdat
	.p2align	3
	.type	DW.ref.__gxx_personality_v0,@object
	.size	DW.ref.__gxx_personality_v0, 8
DW.ref.__gxx_personality_v0:
	.quad	__gxx_personality_v0
	.ident	"clang version 12.0.0 (git@github.com:llvm/llvm-project.git afd483e57d166418e94a65bd9716e7dc4c114eed)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym _ZTIi
