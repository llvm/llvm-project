# REQUIRES: asserts
# UNSUPPORTED: system-windows
# RUN: llvm-mc -triple=x86_64-pc-linux-gnu -large-code-model \
# RUN:   -filetype=obj -o %t %s
# RUN: llvm-jitlink -debug-only=jitlink -noexec -phony-externals %t 2>&1 | \
# RUN:   FileCheck %s
#
# Check handling of pointer encodings for personality functions when compiling
# with `-mcmodel=large -static`.
#

# CHECK: Record is CIE
# CHECK-NEXT: edge at {{.*}} to personality at {{.*}} (DW.ref.__gxx_personality_v0)
# CHECK: Record is CIE
# CHECK-NEXT: edge at {{.*}} to personality at {{.*}} (__gxx_personality_v0)

        .text
        .file   "eh.cpp"

        .globl main
        .p2align        4, 0x90
        .type   main,@function
main:
        xorl    %eax, %eax
        retq
.Lfunc_end_main:
        .size   main, .Lfunc_end_main-main

# pe_absptr uses absptr encoding for __gxx_personality_v0
	.text
	.globl	pe_absptr
	.p2align	4, 0x90
	.type	pe_absptr,@function
pe_absptr:
.Lfunc_begin0:
	.cfi_startproc
	.cfi_personality 0, __gxx_personality_v0
	.cfi_lsda 0, .Lexception0
	pushq	%rax
	.cfi_def_cfa_offset 16
	movabsq	$__cxa_allocate_exception, %rax
	movl	$4, %edi
	callq	*%rax
	movl	$42, (%rax)
.Ltmp0:
	movabsq	$_ZTIi, %rsi
	movabsq	$__cxa_throw, %rcx
	movq	%rax, %rdi
	xorl	%edx, %edx
	callq	*%rcx
.Ltmp1:
.LBB0_2:
.Ltmp2:
	movabsq	$__cxa_begin_catch, %rcx
	movq	%rax, %rdi
	callq	*%rcx
	movabsq	$__cxa_end_catch, %rax
	popq	%rcx
	.cfi_def_cfa_offset 8
	jmpq	*%rax
.Lfunc_end0:
	.size	pe_absptr, .Lfunc_end0-pe_absptr
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2
GCC_except_table0:
.Lexception0:
	.byte	255                             # @LPStart Encoding = omit
	.byte	0                               # @TType Encoding = absptr
	.uleb128 .Lttbase0-.Lttbaseref0
.Lttbaseref0:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end0-.Lcst_begin0
.Lcst_begin0:
	.uleb128 .Lfunc_begin0-.Lfunc_begin0    # >> Call Site 1 <<
	.uleb128 .Ltmp0-.Lfunc_begin0           #   Call between .Lfunc_begin0 and .Ltmp0
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp0-.Lfunc_begin0           # >> Call Site 2 <<
	.uleb128 .Ltmp1-.Ltmp0                  #   Call between .Ltmp0 and .Ltmp1
	.uleb128 .Ltmp2-.Lfunc_begin0           #     jumps to .Ltmp2
	.byte	1                               #   On action: 1
	.uleb128 .Ltmp1-.Lfunc_begin0           # >> Call Site 3 <<
	.uleb128 .Lfunc_end0-.Ltmp1             #   Call between .Ltmp1 and .Lfunc_end0
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end0:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2
                                        # >> Catch TypeInfos <<
	.quad	_ZTIi                           # TypeInfo 1
.Lttbase0:
	.p2align	2
                                        # -- End function

# pe_indir_pcrel_sdata8 uses 0x9C -- Indirect, pc-rel, sdata8 encoding to
# DW.ref.__gxx_personality_v0
	.text
	.globl	pe_indir_pcrel_sdata8
	.p2align	4, 0x90
	.type	pe_indir_pcrel_sdata8,@function
pe_indir_pcrel_sdata8:
.Lfunc_begin1:
	.cfi_startproc
	.cfi_personality 156, DW.ref.__gxx_personality_v0
	.cfi_lsda 28, .Lexception1
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	pushq	%rax
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
.L1$pb:
	leaq	.L1$pb(%rip), %rax
	movabsq	$_GLOBAL_OFFSET_TABLE_-.L1$pb, %rbx
	addq	%rax, %rbx
	movabsq	$__cxa_allocate_exception@GOT, %rax
	movl	$4, %edi
	callq	*(%rbx,%rax)
	movl	$42, (%rax)
.Ltmp4:
	movabsq	$_ZTIi@GOT, %rcx
	movq	(%rbx,%rcx), %rsi
	movabsq	$__cxa_throw@GOT, %rcx
	movq	%rax, %rdi
	xorl	%edx, %edx
	movq	%rbx, %r14
	callq	*(%rbx,%rcx)
.Ltmp5:
.LBB1_2:
.Ltmp6:
	movabsq	$__cxa_begin_catch@GOT, %rcx
	movq	%rax, %rdi
	callq	*(%r14,%rcx)
	movabsq	$__cxa_end_catch@GOT, %rax
	movq	%r14, %rcx
	addq	$8, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	jmpq	*(%rcx,%rax)
.Lfunc_end1:
	.size	pe_indir_pcrel_sdata8, .Lfunc_end1-pe_indir_pcrel_sdata8
	.cfi_endproc
	.section	.gcc_except_table,"a",@progbits
	.p2align	2
GCC_except_table1:
.Lexception1:
	.byte	255                             # @LPStart Encoding = omit
	.byte	156                             # @TType Encoding = indirect pcrel sdata8
	.uleb128 .Lttbase1-.Lttbaseref1
.Lttbaseref1:
	.byte	1                               # Call site Encoding = uleb128
	.uleb128 .Lcst_end1-.Lcst_begin1
.Lcst_begin1:
	.uleb128 .Lfunc_begin1-.Lfunc_begin1    # >> Call Site 1 <<
	.uleb128 .Ltmp4-.Lfunc_begin1           #   Call between .Lfunc_begin1 and .Ltmp4
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
	.uleb128 .Ltmp4-.Lfunc_begin1           # >> Call Site 2 <<
	.uleb128 .Ltmp5-.Ltmp4                  #   Call between .Ltmp4 and .Ltmp5
	.uleb128 .Ltmp6-.Lfunc_begin1           #     jumps to .Ltmp6
	.byte	1                               #   On action: 1
	.uleb128 .Ltmp5-.Lfunc_begin1           # >> Call Site 3 <<
	.uleb128 .Lfunc_end1-.Ltmp5             #   Call between .Ltmp5 and .Lfunc_end1
	.byte	0                               #     has no landing pad
	.byte	0                               #   On action: cleanup
.Lcst_end1:
	.byte	1                               # >> Action Record 1 <<
                                        #   Catch TypeInfo 1
	.byte	0                               #   No further actions
	.p2align	2
                                        # >> Catch TypeInfos <<
.Ltmp7:                                 # TypeInfo 1
	.quad	.L_ZTIi.DW.stub-.Ltmp7
.Lttbase1:
	.p2align	2
                                        # -- End function

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

	.ident	"clang version 13.0.1"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __gxx_personality_v0
	.addrsig_sym _ZTIi
