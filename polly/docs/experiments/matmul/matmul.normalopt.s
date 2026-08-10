	.text
	.file	"matmul.c"
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3               # -- Begin function init_array
.LCPI0_0:
	.quad	4602678819172646912     # double 0.5
	.text
	.globl	init_array
	.p2align	4, 0x90
	.type	init_array,@function
init_array:                             # @init_array
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	leaq	B(%rip), %rax
	leaq	A(%rip), %rcx
	xorl	%r8d, %r8d
	movsd	.LCPI0_0(%rip), %xmm0   # xmm0 = mem[0],zero
	xorl	%r9d, %r9d
	.p2align	4, 0x90
.LBB0_1:                                # %for.cond1.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
	movl	$1, %edi
	xorl	%edx, %edx
	.p2align	4, 0x90
.LBB0_2:                                # %for.body3
                                        #   Parent Loop BB0_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%edx, %esi
	andl	$1022, %esi             # imm = 0x3FE
	orl	$1, %esi
	xorps	%xmm1, %xmm1
	cvtsi2sdl	%esi, %xmm1
	mulsd	%xmm0, %xmm1
	cvtsd2ss	%xmm1, %xmm1
	movss	%xmm1, -4(%rcx,%rdi,4)
	movss	%xmm1, -4(%rax,%rdi,4)
	leal	(%r9,%rdx), %esi
	andl	$1023, %esi             # imm = 0x3FF
	addl	$1, %esi
	xorps	%xmm1, %xmm1
	cvtsi2sdl	%esi, %xmm1
	mulsd	%xmm0, %xmm1
	cvtsd2ss	%xmm1, %xmm1
	movss	%xmm1, (%rcx,%rdi,4)
	movss	%xmm1, (%rax,%rdi,4)
	addq	$2, %rdi
	addl	%r8d, %edx
	cmpq	$1537, %rdi             # imm = 0x601
	jne	.LBB0_2
# %bb.3:                                # %for.inc17
                                        #   in Loop: Header=BB0_1 Depth=1
	addq	$1, %r9
	addq	$6144, %rax             # imm = 0x1800
	addq	$6144, %rcx             # imm = 0x1800
	addl	$2, %r8d
	cmpq	$1536, %r9              # imm = 0x600
	jne	.LBB0_1
# %bb.4:                                # %for.end19
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end0:
	.size	init_array, .Lfunc_end0-init_array
	.cfi_endproc
                                        # -- End function
	.globl	print_array             # -- Begin function print_array
	.p2align	4, 0x90
	.type	print_array,@function
print_array:                            # @print_array
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	pushq	%rax
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	leaq	C(%rip), %r13
	xorl	%eax, %eax
	movl	$3435973837, %r12d      # imm = 0xCCCCCCCD
	leaq	.L.str(%rip), %r14
	.p2align	4, 0x90
.LBB1_1:                                # %for.cond1.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_2 Depth 2
	movq	%rax, -48(%rbp)         # 8-byte Spill
	movq	stdout(%rip), %rsi
	xorl	%ebx, %ebx
	.p2align	4, 0x90
.LBB1_2:                                # %for.body3
                                        #   Parent Loop BB1_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%ebx, %eax
	imulq	%r12, %rax
	shrq	$38, %rax
	leal	(%rax,%rax,4), %r15d
	shll	$4, %r15d
	addl	$79, %r15d
	movss	(%r13,%rbx,4), %xmm0    # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	movb	$1, %al
	movq	%rsi, %rdi
	movq	%r14, %rsi
	callq	fprintf
	cmpl	%ebx, %r15d
	jne	.LBB1_4
# %bb.3:                                # %if.then
                                        #   in Loop: Header=BB1_2 Depth=2
	movq	stdout(%rip), %rsi
	movl	$10, %edi
	callq	fputc@PLT
.LBB1_4:                                # %for.inc
                                        #   in Loop: Header=BB1_2 Depth=2
	addq	$1, %rbx
	movq	stdout(%rip), %rsi
	cmpq	$1536, %rbx             # imm = 0x600
	jne	.LBB1_2
# %bb.5:                                # %for.end
                                        #   in Loop: Header=BB1_1 Depth=1
	movl	$10, %edi
	callq	fputc@PLT
	movq	-48(%rbp), %rax         # 8-byte Reload
	addq	$1, %rax
	addq	$6144, %r13             # imm = 0x1800
	cmpq	$1536, %rax             # imm = 0x600
	jne	.LBB1_1
# %bb.6:                                # %for.end12
	addq	$8, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end1:
	.size	print_array, .Lfunc_end1-print_array
	.cfi_endproc
                                        # -- End function
	.globl	main                    # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:                                # %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
	callq	init_array
	leaq	A(%rip), %rax
	xorl	%r10d, %r10d
	leaq	B(%rip), %r8
	leaq	C(%rip), %r9
	.p2align	4, 0x90
.LBB2_1:                                # %for.cond1.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB2_2 Depth 2
                                        #       Child Loop BB2_3 Depth 3
	movq	%r8, %rsi
	xorl	%edx, %edx
	.p2align	4, 0x90
.LBB2_2:                                # %for.body3
                                        #   Parent Loop BB2_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB2_3 Depth 3
	leaq	(%r10,%r10,2), %rcx
	shlq	$11, %rcx
	addq	%r9, %rcx
	leaq	(%rcx,%rdx,4), %r11
	movl	$0, (%rcx,%rdx,4)
	xorps	%xmm0, %xmm0
	movl	$2, %ecx
	movq	%rsi, %rdi
	.p2align	4, 0x90
.LBB2_3:                                # %for.body8
                                        #   Parent Loop BB2_1 Depth=1
                                        #     Parent Loop BB2_2 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movss	-8(%rax,%rcx,4), %xmm1  # xmm1 = mem[0],zero,zero,zero
	mulss	(%rdi), %xmm1
	movss	-4(%rax,%rcx,4), %xmm2  # xmm2 = mem[0],zero,zero,zero
	addss	%xmm0, %xmm1
	mulss	6144(%rdi), %xmm2
	addss	%xmm1, %xmm2
	movss	(%rax,%rcx,4), %xmm0    # xmm0 = mem[0],zero,zero,zero
	mulss	12288(%rdi), %xmm0
	addss	%xmm2, %xmm0
	addq	$3, %rcx
	addq	$18432, %rdi            # imm = 0x4800
	cmpq	$1538, %rcx             # imm = 0x602
	jne	.LBB2_3
# %bb.4:                                # %for.inc25
                                        #   in Loop: Header=BB2_2 Depth=2
	movss	%xmm0, (%r11)
	addq	$1, %rdx
	addq	$4, %rsi
	cmpq	$1536, %rdx             # imm = 0x600
	jne	.LBB2_2
# %bb.5:                                # %for.inc28
                                        #   in Loop: Header=BB2_1 Depth=1
	addq	$1, %r10
	addq	$6144, %rax             # imm = 0x1800
	cmpq	$1536, %r10             # imm = 0x600
	jne	.LBB2_1
# %bb.6:                                # %for.end30
	xorl	%eax, %eax
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end2:
	.size	main, .Lfunc_end2-main
	.cfi_endproc
                                        # -- End function
	.type	A,@object               # @A
	.comm	A,9437184,16
	.type	B,@object               # @B
	.comm	B,9437184,16
	.type	.L.str,@object          # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"%lf "
	.size	.L.str, 5

	.type	C,@object               # @C
	.comm	C,9437184,16

	.ident	"clang version 8.0.0 (trunk 342834) (llvm/trunk 342856)"
	.section	".note.GNU-stack","",@progbits
