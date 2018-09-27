	.text
	.file	"matmul.c"
	.globl	init_array              # -- Begin function init_array
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
	pushq	%rbx
	pushq	%rax
	.cfi_offset %rbx, -24
	leaq	init_array_polly_subfn(%rip), %rdi
	leaq	-16(%rbp), %rbx
	xorl	%edx, %edx
	xorl	%ecx, %ecx
	movl	$1536, %r8d             # imm = 0x600
	movl	$1, %r9d
	movq	%rbx, %rsi
	callq	GOMP_parallel_loop_runtime_start@PLT
	movq	%rbx, %rdi
	callq	init_array_polly_subfn
	callq	GOMP_parallel_end@PLT
	addq	$8, %rsp
	popq	%rbx
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
	pushq	%rbx
	pushq	%rax
	.cfi_offset %rbx, -24
	callq	init_array
	leaq	main_polly_subfn(%rip), %rdi
	leaq	-16(%rbp), %rbx
	xorl	%edx, %edx
	xorl	%ecx, %ecx
	movl	$1536, %r8d             # imm = 0x600
	movl	$1, %r9d
	movq	%rbx, %rsi
	callq	GOMP_parallel_loop_runtime_start@PLT
	movq	%rbx, %rdi
	callq	main_polly_subfn
	callq	GOMP_parallel_end@PLT
	leaq	main_polly_subfn_1(%rip), %rdi
	xorl	%edx, %edx
	xorl	%ecx, %ecx
	movl	$1536, %r8d             # imm = 0x600
	movl	$64, %r9d
	movq	%rbx, %rsi
	callq	GOMP_parallel_loop_runtime_start@PLT
	movq	%rbx, %rdi
	callq	main_polly_subfn_1
	callq	GOMP_parallel_end@PLT
	xorl	%eax, %eax
	addq	$8, %rsp
	popq	%rbx
	popq	%rbp
	.cfi_def_cfa %rsp, 8
	retq
.Lfunc_end2:
	.size	main, .Lfunc_end2-main
	.cfi_endproc
                                        # -- End function
	.section	.rodata.cst8,"aM",@progbits,8
	.p2align	3               # -- Begin function init_array_polly_subfn
.LCPI3_0:
	.quad	4602678819172646912     # double 0.5
	.text
	.p2align	4, 0x90
	.type	init_array_polly_subfn,@function
init_array_polly_subfn:                 # @init_array_polly_subfn
	.cfi_startproc
# %bb.0:                                # %polly.par.setup
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r13
	.cfi_def_cfa_offset 32
	pushq	%r12
	.cfi_def_cfa_offset 40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	subq	$16, %rsp
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r13, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	leaq	8(%rsp), %rdi
	movq	%rsp, %rsi
	callq	GOMP_loop_runtime_next@PLT
	testb	%al, %al
	je	.LBB3_2
# %bb.1:
	leaq	B(%rip), %r15
	leaq	A(%rip), %r12
	movsd	.LCPI3_0(%rip), %xmm1   # xmm1 = mem[0],zero
	leaq	8(%rsp), %r14
	movq	%rsp, %r13
	.p2align	4, 0x90
.LBB3_4:                                # %polly.par.loadIVBounds
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_5 Depth 2
                                        #       Child Loop BB3_6 Depth 3
	movq	8(%rsp), %rax
	movq	(%rsp), %r8
	decq	%r8
	movq	%rax, %rdx
	shlq	$11, %rdx
	leaq	(%rdx,%rdx,2), %rdx
	leaq	(%r15,%rdx), %rsi
	addq	%r12, %rdx
	.p2align	4, 0x90
.LBB3_5:                                # %polly.loop_header
                                        #   Parent Loop BB3_4 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB3_6 Depth 3
	movq	$-6144, %rdi            # imm = 0xE800
	xorl	%ecx, %ecx
	.p2align	4, 0x90
.LBB3_6:                                # %polly.loop_header2
                                        #   Parent Loop BB3_4 Depth=1
                                        #     Parent Loop BB3_5 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movl	%ecx, %ebx
	andl	$1023, %ebx             # imm = 0x3FF
	incl	%ebx
	xorps	%xmm0, %xmm0
	cvtsi2sdl	%ebx, %xmm0
	mulsd	%xmm1, %xmm0
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, 6144(%rdx,%rdi)
	movss	%xmm0, 6144(%rsi,%rdi)
	addl	%eax, %ecx
	addq	$4, %rdi
	jne	.LBB3_6
# %bb.7:                                # %polly.loop_exit4
                                        #   in Loop: Header=BB3_5 Depth=2
	addq	$6144, %rsi             # imm = 0x1800
	addq	$6144, %rdx             # imm = 0x1800
	cmpq	%r8, %rax
	leaq	1(%rax), %rax
	jl	.LBB3_5
# %bb.3:                                # %polly.par.checkNext.loopexit
                                        #   in Loop: Header=BB3_4 Depth=1
	movq	%r14, %rdi
	movq	%r13, %rsi
	callq	GOMP_loop_runtime_next@PLT
	movsd	.LCPI3_0(%rip), %xmm1   # xmm1 = mem[0],zero
	testb	%al, %al
	jne	.LBB3_4
.LBB3_2:                                # %polly.par.exit
	callq	GOMP_loop_end_nowait@PLT
	addq	$16, %rsp
	.cfi_def_cfa_offset 48
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end3:
	.size	init_array_polly_subfn, .Lfunc_end3-init_array_polly_subfn
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90         # -- Begin function main_polly_subfn
	.type	main_polly_subfn,@function
main_polly_subfn:                       # @main_polly_subfn
	.cfi_startproc
# %bb.0:                                # %polly.par.setup
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$16, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	leaq	8(%rsp), %rdi
	movq	%rsp, %rsi
	callq	GOMP_loop_runtime_next@PLT
	testb	%al, %al
	je	.LBB4_3
# %bb.1:
	leaq	C(%rip), %r15
	leaq	8(%rsp), %r14
	movq	%rsp, %rbx
	.p2align	4, 0x90
.LBB4_2:                                # %polly.par.loadIVBounds
                                        # =>This Inner Loop Header: Depth=1
	movq	8(%rsp), %rax
	movq	(%rsp), %rcx
	decq	%rcx
	leaq	(%rax,%rax,2), %rdi
	shlq	$11, %rdi
	addq	%r15, %rdi
	cmpq	%rcx, %rax
	cmovgeq	%rax, %rcx
	incq	%rcx
	subq	%rax, %rcx
	shlq	$11, %rcx
	leaq	(%rcx,%rcx,2), %rdx
	xorl	%esi, %esi
	callq	memset@PLT
	movq	%r14, %rdi
	movq	%rbx, %rsi
	callq	GOMP_loop_runtime_next@PLT
	testb	%al, %al
	jne	.LBB4_2
.LBB4_3:                                # %polly.par.exit
	callq	GOMP_loop_end_nowait@PLT
	addq	$16, %rsp
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end4:
	.size	main_polly_subfn, .Lfunc_end4-main_polly_subfn
	.cfi_endproc
                                        # -- End function
	.p2align	4, 0x90         # -- Begin function main_polly_subfn_1
	.type	main_polly_subfn_1,@function
main_polly_subfn_1:                     # @main_polly_subfn_1
	.cfi_startproc
# %bb.0:                                # %polly.par.setup
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r15
	.cfi_def_cfa_offset 24
	pushq	%r14
	.cfi_def_cfa_offset 32
	pushq	%r13
	.cfi_def_cfa_offset 40
	pushq	%r12
	.cfi_def_cfa_offset 48
	pushq	%rbx
	.cfi_def_cfa_offset 56
	subq	$296, %rsp              # imm = 0x128
	.cfi_def_cfa_offset 352
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	.cfi_offset %rbp, -16
	jmp	.LBB5_1
	.p2align	4, 0x90
.LBB5_2:                                # %polly.par.loadIVBounds
                                        #   in Loop: Header=BB5_1 Depth=1
	movq	40(%rsp), %rdx
	movq	32(%rsp), %rax
	decq	%rax
	movq	%rax, 136(%rsp)         # 8-byte Spill
	leaq	(%rdx,%rdx,2), %rcx
	shlq	$11, %rcx
	leaq	A(%rip), %rax
	addq	%rax, %rcx
	movq	%rcx, 24(%rsp)          # 8-byte Spill
	.p2align	4, 0x90
.LBB5_3:                                # %polly.loop_header
                                        #   Parent Loop BB5_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB5_4 Depth 3
                                        #         Child Loop BB5_5 Depth 4
                                        #           Child Loop BB5_6 Depth 5
                                        #             Child Loop BB5_7 Depth 6
	leaq	63(%rdx), %rsi
	leaq	B+192(%rip), %r14
	xorl	%ecx, %ecx
	xorl	%eax, %eax
	movq	%rdx, 168(%rsp)         # 8-byte Spill
	.p2align	4, 0x90
.LBB5_4:                                # %polly.loop_header2
                                        #   Parent Loop BB5_1 Depth=1
                                        #     Parent Loop BB5_3 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB5_5 Depth 4
                                        #           Child Loop BB5_6 Depth 5
                                        #             Child Loop BB5_7 Depth 6
	movq	%rax, 144(%rsp)         # 8-byte Spill
	movq	%rcx, 152(%rsp)         # 8-byte Spill
	shlq	$6, %rcx
	leaq	16(%rcx), %rdi
	leaq	32(%rcx), %rbp
	leaq	48(%rcx), %r15
	movq	24(%rsp), %r9           # 8-byte Reload
	movq	%r14, 160(%rsp)         # 8-byte Spill
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB5_5:                                # %polly.loop_header8
                                        #   Parent Loop BB5_1 Depth=1
                                        #     Parent Loop BB5_3 Depth=2
                                        #       Parent Loop BB5_4 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB5_6 Depth 5
                                        #             Child Loop BB5_7 Depth 6
	movq	%rax, 176(%rsp)         # 8-byte Spill
	movq	%r9, 184(%rsp)          # 8-byte Spill
	movq	%rdx, %rax
	.p2align	4, 0x90
.LBB5_6:                                # %polly.loop_header14
                                        #   Parent Loop BB5_1 Depth=1
                                        #     Parent Loop BB5_3 Depth=2
                                        #       Parent Loop BB5_4 Depth=3
                                        #         Parent Loop BB5_5 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB5_7 Depth 6
	leaq	(%rax,%rax,2), %rbx
	shlq	$11, %rbx
	leaq	C(%rip), %rdx
	addq	%rdx, %rbx
	leaq	(%rbx,%rcx,4), %r8
	leaq	(%rbx,%rdi,4), %rdx
	leaq	(%rbx,%rbp,4), %r13
	leaq	(%rbx,%r15,4), %r10
	movups	(%rbx,%rcx,4), %xmm8
	movups	16(%rbx,%rcx,4), %xmm0
	movaps	%xmm0, 96(%rsp)         # 16-byte Spill
	movups	32(%rbx,%rcx,4), %xmm6
	movups	48(%rbx,%rcx,4), %xmm1
	movups	(%rbx,%rdi,4), %xmm15
	movups	16(%rbx,%rdi,4), %xmm0
	movaps	%xmm0, (%rsp)           # 16-byte Spill
	movups	32(%rbx,%rdi,4), %xmm0
	movaps	%xmm0, 48(%rsp)         # 16-byte Spill
	movups	48(%rbx,%rdi,4), %xmm0
	movaps	%xmm0, 64(%rsp)         # 16-byte Spill
	movups	(%rbx,%rbp,4), %xmm11
	movups	16(%rbx,%rbp,4), %xmm0
	movaps	%xmm0, 112(%rsp)        # 16-byte Spill
	movups	32(%rbx,%rbp,4), %xmm12
	movups	48(%rbx,%rbp,4), %xmm0
	movaps	%xmm0, 80(%rsp)         # 16-byte Spill
	movups	(%rbx,%r15,4), %xmm9
	movups	16(%rbx,%r15,4), %xmm13
	movups	32(%rbx,%r15,4), %xmm2
	movups	48(%rbx,%r15,4), %xmm3
	movq	$-256, %r12
	movq	%r14, %r11
	.p2align	4, 0x90
.LBB5_7:                                # %vector.ph
                                        #   Parent Loop BB5_1 Depth=1
                                        #     Parent Loop BB5_3 Depth=2
                                        #       Parent Loop BB5_4 Depth=3
                                        #         Parent Loop BB5_5 Depth=4
                                        #           Parent Loop BB5_6 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	movaps	%xmm12, 208(%rsp)       # 16-byte Spill
	movaps	%xmm2, 224(%rsp)        # 16-byte Spill
	movaps	%xmm3, 240(%rsp)        # 16-byte Spill
	movaps	%xmm8, %xmm10
	movaps	96(%rsp), %xmm7         # 16-byte Reload
	unpcklps	%xmm7, %xmm10   # xmm10 = xmm10[0],xmm7[0],xmm10[1],xmm7[1]
	movaps	%xmm1, %xmm4
	shufps	$0, %xmm6, %xmm4        # xmm4 = xmm4[0,0],xmm6[0,0]
	shufps	$36, %xmm4, %xmm10      # xmm10 = xmm10[0,1],xmm4[2,0]
	movaps	%xmm7, %xmm5
	shufps	$17, %xmm8, %xmm5       # xmm5 = xmm5[1,0],xmm8[1,0]
	movaps	%xmm6, %xmm4
	unpcklps	%xmm1, %xmm4    # xmm4 = xmm4[0],xmm1[0],xmm4[1],xmm1[1]
	shufps	$226, %xmm4, %xmm5      # xmm5 = xmm5[2,0],xmm4[2,3]
	movaps	%xmm8, %xmm12
	unpckhps	%xmm7, %xmm12   # xmm12 = xmm12[2],xmm7[2],xmm12[3],xmm7[3]
	movaps	%xmm1, %xmm4
	shufps	$34, %xmm6, %xmm4       # xmm4 = xmm4[2,0],xmm6[2,0]
	shufps	$36, %xmm4, %xmm12      # xmm12 = xmm12[0,1],xmm4[2,0]
	shufps	$51, %xmm8, %xmm7       # xmm7 = xmm7[3,0],xmm8[3,0]
	unpckhps	%xmm1, %xmm6    # xmm6 = xmm6[2],xmm1[2],xmm6[3],xmm1[3]
	shufps	$226, %xmm6, %xmm7      # xmm7 = xmm7[2,0],xmm6[2,3]
	movaps	-160(%r11), %xmm0
	movaps	-144(%r11), %xmm1
	movaps	%xmm1, %xmm6
	shufps	$0, %xmm0, %xmm6        # xmm6 = xmm6[0,0],xmm0[0,0]
	movaps	-192(%r11), %xmm3
	movaps	-176(%r11), %xmm4
	movaps	%xmm3, %xmm8
	unpcklps	%xmm4, %xmm8    # xmm8 = xmm8[0],xmm4[0],xmm8[1],xmm4[1]
	shufps	$36, %xmm6, %xmm8       # xmm8 = xmm8[0,1],xmm6[2,0]
	movaps	%xmm0, %xmm2
	unpcklps	%xmm1, %xmm2    # xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1]
	movaps	%xmm4, %xmm6
	shufps	$17, %xmm3, %xmm6       # xmm6 = xmm6[1,0],xmm3[1,0]
	shufps	$226, %xmm2, %xmm6      # xmm6 = xmm6[2,0],xmm2[2,3]
	movaps	%xmm1, %xmm2
	shufps	$34, %xmm0, %xmm2       # xmm2 = xmm2[2,0],xmm0[2,0]
	movaps	%xmm3, %xmm14
	unpckhps	%xmm4, %xmm14   # xmm14 = xmm14[2],xmm4[2],xmm14[3],xmm4[3]
	shufps	$36, %xmm2, %xmm14      # xmm14 = xmm14[0,1],xmm2[2,0]
	unpckhps	%xmm1, %xmm0    # xmm0 = xmm0[2],xmm1[2],xmm0[3],xmm1[3]
	shufps	$51, %xmm3, %xmm4       # xmm4 = xmm4[3,0],xmm3[3,0]
	shufps	$226, %xmm0, %xmm4      # xmm4 = xmm4[2,0],xmm0[2,3]
	movss	256(%r9,%r12), %xmm0    # xmm0 = mem[0],zero,zero,zero
	shufps	$0, %xmm0, %xmm0        # xmm0 = xmm0[0,0,0,0]
	mulps	%xmm0, %xmm8
	addps	%xmm10, %xmm8
	mulps	%xmm0, %xmm6
	addps	%xmm5, %xmm6
	mulps	%xmm0, %xmm14
	addps	%xmm12, %xmm14
	mulps	%xmm0, %xmm4
	movaps	%xmm0, %xmm5
	addps	%xmm7, %xmm4
	movaps	%xmm14, %xmm0
	unpckhps	%xmm4, %xmm0    # xmm0 = xmm0[2],xmm4[2],xmm0[3],xmm4[3]
	movaps	%xmm6, %xmm1
	shufps	$51, %xmm8, %xmm1       # xmm1 = xmm1[3,0],xmm8[3,0]
	shufps	$226, %xmm0, %xmm1      # xmm1 = xmm1[2,0],xmm0[2,3]
	movaps	%xmm1, 272(%rsp)        # 16-byte Spill
	movaps	%xmm4, %xmm0
	shufps	$34, %xmm14, %xmm0      # xmm0 = xmm0[2,0],xmm14[2,0]
	movaps	%xmm8, %xmm1
	unpckhps	%xmm6, %xmm1    # xmm1 = xmm1[2],xmm6[2],xmm1[3],xmm6[3]
	shufps	$36, %xmm0, %xmm1       # xmm1 = xmm1[0,1],xmm0[2,0]
	movaps	%xmm1, 256(%rsp)        # 16-byte Spill
	movaps	%xmm14, %xmm0
	unpcklps	%xmm4, %xmm0    # xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1]
	movaps	%xmm6, %xmm1
	shufps	$17, %xmm8, %xmm1       # xmm1 = xmm1[1,0],xmm8[1,0]
	shufps	$226, %xmm0, %xmm1      # xmm1 = xmm1[2,0],xmm0[2,3]
	movaps	%xmm1, 96(%rsp)         # 16-byte Spill
	shufps	$0, %xmm14, %xmm4       # xmm4 = xmm4[0,0],xmm14[0,0]
	unpcklps	%xmm6, %xmm8    # xmm8 = xmm8[0],xmm6[0],xmm8[1],xmm6[1]
	shufps	$36, %xmm4, %xmm8       # xmm8 = xmm8[0,1],xmm4[2,0]
	movaps	%xmm15, %xmm14
	movaps	(%rsp), %xmm4           # 16-byte Reload
	unpcklps	%xmm4, %xmm14   # xmm14 = xmm14[0],xmm4[0],xmm14[1],xmm4[1]
	movaps	64(%rsp), %xmm1         # 16-byte Reload
	movaps	%xmm1, %xmm0
	movaps	48(%rsp), %xmm3         # 16-byte Reload
	shufps	$0, %xmm3, %xmm0        # xmm0 = xmm0[0,0],xmm3[0,0]
	shufps	$36, %xmm0, %xmm14      # xmm14 = xmm14[0,1],xmm0[2,0]
	movaps	%xmm4, %xmm12
	shufps	$17, %xmm15, %xmm12     # xmm12 = xmm12[1,0],xmm15[1,0]
	movaps	%xmm3, %xmm2
	unpcklps	%xmm1, %xmm2    # xmm2 = xmm2[0],xmm1[0],xmm2[1],xmm1[1]
	shufps	$226, %xmm2, %xmm12     # xmm12 = xmm12[2,0],xmm2[2,3]
	movaps	%xmm15, %xmm7
	unpckhps	%xmm4, %xmm7    # xmm7 = xmm7[2],xmm4[2],xmm7[3],xmm4[3]
	movaps	%xmm1, %xmm2
	shufps	$34, %xmm3, %xmm2       # xmm2 = xmm2[2,0],xmm3[2,0]
	shufps	$36, %xmm2, %xmm7       # xmm7 = xmm7[0,1],xmm2[2,0]
	shufps	$51, %xmm15, %xmm4      # xmm4 = xmm4[3,0],xmm15[3,0]
	unpckhps	%xmm1, %xmm3    # xmm3 = xmm3[2],xmm1[2],xmm3[3],xmm1[3]
	shufps	$226, %xmm3, %xmm4      # xmm4 = xmm4[2,0],xmm3[2,3]
	movaps	%xmm4, (%rsp)           # 16-byte Spill
	movaps	-96(%r11), %xmm2
	movaps	-80(%r11), %xmm1
	movaps	%xmm1, %xmm4
	shufps	$0, %xmm2, %xmm4        # xmm4 = xmm4[0,0],xmm2[0,0]
	movaps	-112(%r11), %xmm10
	movaps	-128(%r11), %xmm0
	movaps	%xmm0, %xmm15
	unpcklps	%xmm10, %xmm15  # xmm15 = xmm15[0],xmm10[0],xmm15[1],xmm10[1]
	shufps	$36, %xmm4, %xmm15      # xmm15 = xmm15[0,1],xmm4[2,0]
	movaps	%xmm2, %xmm4
	unpcklps	%xmm1, %xmm4    # xmm4 = xmm4[0],xmm1[0],xmm4[1],xmm1[1]
	movaps	%xmm10, %xmm6
	shufps	$17, %xmm0, %xmm6       # xmm6 = xmm6[1,0],xmm0[1,0]
	shufps	$226, %xmm4, %xmm6      # xmm6 = xmm6[2,0],xmm4[2,3]
	movaps	%xmm1, %xmm3
	shufps	$34, %xmm2, %xmm3       # xmm3 = xmm3[2,0],xmm2[2,0]
	movaps	%xmm0, %xmm4
	unpckhps	%xmm10, %xmm4   # xmm4 = xmm4[2],xmm10[2],xmm4[3],xmm10[3]
	shufps	$36, %xmm3, %xmm4       # xmm4 = xmm4[0,1],xmm3[2,0]
	unpckhps	%xmm1, %xmm2    # xmm2 = xmm2[2],xmm1[2],xmm2[3],xmm1[3]
	shufps	$51, %xmm0, %xmm10      # xmm10 = xmm10[3,0],xmm0[3,0]
	shufps	$226, %xmm2, %xmm10     # xmm10 = xmm10[2,0],xmm2[2,3]
	movaps	%xmm5, 192(%rsp)        # 16-byte Spill
	mulps	%xmm5, %xmm15
	addps	%xmm14, %xmm15
	mulps	%xmm5, %xmm6
	addps	%xmm12, %xmm6
	mulps	%xmm5, %xmm4
	addps	%xmm7, %xmm4
	mulps	%xmm5, %xmm10
	addps	(%rsp), %xmm10          # 16-byte Folded Reload
	movaps	%xmm4, %xmm0
	unpckhps	%xmm10, %xmm0   # xmm0 = xmm0[2],xmm10[2],xmm0[3],xmm10[3]
	movaps	%xmm6, %xmm1
	shufps	$51, %xmm15, %xmm1      # xmm1 = xmm1[3,0],xmm15[3,0]
	shufps	$226, %xmm0, %xmm1      # xmm1 = xmm1[2,0],xmm0[2,3]
	movaps	%xmm1, 64(%rsp)         # 16-byte Spill
	movaps	%xmm10, %xmm0
	shufps	$34, %xmm4, %xmm0       # xmm0 = xmm0[2,0],xmm4[2,0]
	movaps	%xmm15, %xmm1
	unpckhps	%xmm6, %xmm1    # xmm1 = xmm1[2],xmm6[2],xmm1[3],xmm6[3]
	shufps	$36, %xmm0, %xmm1       # xmm1 = xmm1[0,1],xmm0[2,0]
	movaps	%xmm1, 48(%rsp)         # 16-byte Spill
	movaps	%xmm4, %xmm0
	unpcklps	%xmm10, %xmm0   # xmm0 = xmm0[0],xmm10[0],xmm0[1],xmm10[1]
	movaps	%xmm6, %xmm1
	shufps	$17, %xmm15, %xmm1      # xmm1 = xmm1[1,0],xmm15[1,0]
	shufps	$226, %xmm0, %xmm1      # xmm1 = xmm1[2,0],xmm0[2,3]
	movaps	%xmm1, (%rsp)           # 16-byte Spill
	shufps	$0, %xmm4, %xmm10       # xmm10 = xmm10[0,0],xmm4[0,0]
	unpcklps	%xmm6, %xmm15   # xmm15 = xmm15[0],xmm6[0],xmm15[1],xmm6[1]
	shufps	$36, %xmm10, %xmm15     # xmm15 = xmm15[0,1],xmm10[2,0]
	movaps	%xmm11, %xmm10
	movaps	112(%rsp), %xmm14       # 16-byte Reload
	unpcklps	%xmm14, %xmm10  # xmm10 = xmm10[0],xmm14[0],xmm10[1],xmm14[1]
	movaps	80(%rsp), %xmm2         # 16-byte Reload
	movaps	%xmm2, %xmm0
	movaps	208(%rsp), %xmm3        # 16-byte Reload
	shufps	$0, %xmm3, %xmm0        # xmm0 = xmm0[0,0],xmm3[0,0]
	shufps	$36, %xmm0, %xmm10      # xmm10 = xmm10[0,1],xmm0[2,0]
	movaps	%xmm14, %xmm12
	shufps	$17, %xmm11, %xmm12     # xmm12 = xmm12[1,0],xmm11[1,0]
	movaps	%xmm3, %xmm0
	unpcklps	%xmm2, %xmm0    # xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
	shufps	$226, %xmm0, %xmm12     # xmm12 = xmm12[2,0],xmm0[2,3]
	movaps	%xmm11, %xmm0
	unpckhps	%xmm14, %xmm0   # xmm0 = xmm0[2],xmm14[2],xmm0[3],xmm14[3]
	movaps	%xmm2, %xmm1
	shufps	$34, %xmm3, %xmm1       # xmm1 = xmm1[2,0],xmm3[2,0]
	shufps	$36, %xmm1, %xmm0       # xmm0 = xmm0[0,1],xmm1[2,0]
	shufps	$51, %xmm11, %xmm14     # xmm14 = xmm14[3,0],xmm11[3,0]
	unpckhps	%xmm2, %xmm3    # xmm3 = xmm3[2],xmm2[2],xmm3[3],xmm2[3]
	shufps	$226, %xmm3, %xmm14     # xmm14 = xmm14[2,0],xmm3[2,3]
	movaps	-32(%r11), %xmm1
	movaps	-16(%r11), %xmm2
	movaps	%xmm2, %xmm3
	shufps	$0, %xmm1, %xmm3        # xmm3 = xmm3[0,0],xmm1[0,0]
	movaps	-48(%r11), %xmm4
	movaps	-64(%r11), %xmm5
	movaps	%xmm5, %xmm11
	unpcklps	%xmm4, %xmm11   # xmm11 = xmm11[0],xmm4[0],xmm11[1],xmm4[1]
	shufps	$36, %xmm3, %xmm11      # xmm11 = xmm11[0,1],xmm3[2,0]
	movaps	%xmm1, %xmm3
	unpcklps	%xmm2, %xmm3    # xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1]
	movaps	%xmm4, %xmm7
	shufps	$17, %xmm5, %xmm7       # xmm7 = xmm7[1,0],xmm5[1,0]
	shufps	$226, %xmm3, %xmm7      # xmm7 = xmm7[2,0],xmm3[2,3]
	movaps	%xmm2, %xmm3
	shufps	$34, %xmm1, %xmm3       # xmm3 = xmm3[2,0],xmm1[2,0]
	movaps	%xmm5, %xmm6
	unpckhps	%xmm4, %xmm6    # xmm6 = xmm6[2],xmm4[2],xmm6[3],xmm4[3]
	shufps	$36, %xmm3, %xmm6       # xmm6 = xmm6[0,1],xmm3[2,0]
	unpckhps	%xmm2, %xmm1    # xmm1 = xmm1[2],xmm2[2],xmm1[3],xmm2[3]
	shufps	$51, %xmm5, %xmm4       # xmm4 = xmm4[3,0],xmm5[3,0]
	shufps	$226, %xmm1, %xmm4      # xmm4 = xmm4[2,0],xmm1[2,3]
	movaps	192(%rsp), %xmm1        # 16-byte Reload
	mulps	%xmm1, %xmm11
	addps	%xmm10, %xmm11
	mulps	%xmm1, %xmm7
	addps	%xmm12, %xmm7
	mulps	%xmm1, %xmm6
	addps	%xmm0, %xmm6
	mulps	%xmm1, %xmm4
	addps	%xmm14, %xmm4
	movaps	%xmm6, %xmm0
	unpckhps	%xmm4, %xmm0    # xmm0 = xmm0[2],xmm4[2],xmm0[3],xmm4[3]
	movaps	%xmm7, %xmm1
	shufps	$51, %xmm11, %xmm1      # xmm1 = xmm1[3,0],xmm11[3,0]
	shufps	$226, %xmm0, %xmm1      # xmm1 = xmm1[2,0],xmm0[2,3]
	movaps	%xmm1, 80(%rsp)         # 16-byte Spill
	movaps	%xmm4, %xmm0
	shufps	$34, %xmm6, %xmm0       # xmm0 = xmm0[2,0],xmm6[2,0]
	movaps	%xmm11, %xmm12
	unpckhps	%xmm7, %xmm12   # xmm12 = xmm12[2],xmm7[2],xmm12[3],xmm7[3]
	shufps	$36, %xmm0, %xmm12      # xmm12 = xmm12[0,1],xmm0[2,0]
	movaps	%xmm6, %xmm0
	unpcklps	%xmm4, %xmm0    # xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1]
	movaps	%xmm7, %xmm1
	shufps	$17, %xmm11, %xmm1      # xmm1 = xmm1[1,0],xmm11[1,0]
	shufps	$226, %xmm0, %xmm1      # xmm1 = xmm1[2,0],xmm0[2,3]
	movaps	%xmm1, 112(%rsp)        # 16-byte Spill
	shufps	$0, %xmm6, %xmm4        # xmm4 = xmm4[0,0],xmm6[0,0]
	unpcklps	%xmm7, %xmm11   # xmm11 = xmm11[0],xmm7[0],xmm11[1],xmm7[1]
	shufps	$36, %xmm4, %xmm11      # xmm11 = xmm11[0,1],xmm4[2,0]
	movaps	%xmm9, %xmm10
	unpcklps	%xmm13, %xmm10  # xmm10 = xmm10[0],xmm13[0],xmm10[1],xmm13[1]
	movaps	240(%rsp), %xmm2        # 16-byte Reload
	movaps	%xmm2, %xmm0
	movaps	224(%rsp), %xmm3        # 16-byte Reload
	shufps	$0, %xmm3, %xmm0        # xmm0 = xmm0[0,0],xmm3[0,0]
	shufps	$36, %xmm0, %xmm10      # xmm10 = xmm10[0,1],xmm0[2,0]
	movaps	%xmm13, %xmm14
	shufps	$17, %xmm9, %xmm14      # xmm14 = xmm14[1,0],xmm9[1,0]
	movaps	%xmm3, %xmm0
	unpcklps	%xmm2, %xmm0    # xmm0 = xmm0[0],xmm2[0],xmm0[1],xmm2[1]
	shufps	$226, %xmm0, %xmm14     # xmm14 = xmm14[2,0],xmm0[2,3]
	movaps	%xmm9, %xmm0
	unpckhps	%xmm13, %xmm0   # xmm0 = xmm0[2],xmm13[2],xmm0[3],xmm13[3]
	movaps	%xmm2, %xmm1
	shufps	$34, %xmm3, %xmm1       # xmm1 = xmm1[2,0],xmm3[2,0]
	shufps	$36, %xmm1, %xmm0       # xmm0 = xmm0[0,1],xmm1[2,0]
	shufps	$51, %xmm9, %xmm13      # xmm13 = xmm13[3,0],xmm9[3,0]
	unpckhps	%xmm2, %xmm3    # xmm3 = xmm3[2],xmm2[2],xmm3[3],xmm2[3]
	shufps	$226, %xmm3, %xmm13     # xmm13 = xmm13[2,0],xmm3[2,3]
	movaps	32(%r11), %xmm1
	movaps	48(%r11), %xmm2
	movaps	%xmm2, %xmm3
	shufps	$0, %xmm1, %xmm3        # xmm3 = xmm3[0,0],xmm1[0,0]
	movaps	16(%r11), %xmm4
	movaps	(%r11), %xmm5
	movaps	%xmm5, %xmm9
	unpcklps	%xmm4, %xmm9    # xmm9 = xmm9[0],xmm4[0],xmm9[1],xmm4[1]
	shufps	$36, %xmm3, %xmm9       # xmm9 = xmm9[0,1],xmm3[2,0]
	movaps	%xmm1, %xmm3
	unpcklps	%xmm2, %xmm3    # xmm3 = xmm3[0],xmm2[0],xmm3[1],xmm2[1]
	movaps	%xmm4, %xmm7
	shufps	$17, %xmm5, %xmm7       # xmm7 = xmm7[1,0],xmm5[1,0]
	shufps	$226, %xmm3, %xmm7      # xmm7 = xmm7[2,0],xmm3[2,3]
	movaps	%xmm2, %xmm3
	shufps	$34, %xmm1, %xmm3       # xmm3 = xmm3[2,0],xmm1[2,0]
	movaps	%xmm5, %xmm6
	unpckhps	%xmm4, %xmm6    # xmm6 = xmm6[2],xmm4[2],xmm6[3],xmm4[3]
	shufps	$36, %xmm3, %xmm6       # xmm6 = xmm6[0,1],xmm3[2,0]
	unpckhps	%xmm2, %xmm1    # xmm1 = xmm1[2],xmm2[2],xmm1[3],xmm2[3]
	shufps	$51, %xmm5, %xmm4       # xmm4 = xmm4[3,0],xmm5[3,0]
	shufps	$226, %xmm1, %xmm4      # xmm4 = xmm4[2,0],xmm1[2,3]
	movaps	192(%rsp), %xmm1        # 16-byte Reload
	mulps	%xmm1, %xmm9
	addps	%xmm10, %xmm9
	mulps	%xmm1, %xmm7
	addps	%xmm14, %xmm7
	mulps	%xmm1, %xmm6
	addps	%xmm0, %xmm6
	mulps	%xmm1, %xmm4
	addps	%xmm13, %xmm4
	movaps	%xmm6, %xmm0
	unpckhps	%xmm4, %xmm0    # xmm0 = xmm0[2],xmm4[2],xmm0[3],xmm4[3]
	movaps	%xmm7, %xmm3
	shufps	$51, %xmm9, %xmm3       # xmm3 = xmm3[3,0],xmm9[3,0]
	shufps	$226, %xmm0, %xmm3      # xmm3 = xmm3[2,0],xmm0[2,3]
	movaps	%xmm4, %xmm0
	shufps	$34, %xmm6, %xmm0       # xmm0 = xmm0[2,0],xmm6[2,0]
	movaps	%xmm9, %xmm2
	unpckhps	%xmm7, %xmm2    # xmm2 = xmm2[2],xmm7[2],xmm2[3],xmm7[3]
	shufps	$36, %xmm0, %xmm2       # xmm2 = xmm2[0,1],xmm0[2,0]
	movaps	%xmm6, %xmm0
	unpcklps	%xmm4, %xmm0    # xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1]
	movaps	%xmm7, %xmm13
	shufps	$17, %xmm9, %xmm13      # xmm13 = xmm13[1,0],xmm9[1,0]
	shufps	$226, %xmm0, %xmm13     # xmm13 = xmm13[2,0],xmm0[2,3]
	shufps	$0, %xmm6, %xmm4        # xmm4 = xmm4[0,0],xmm6[0,0]
	movaps	256(%rsp), %xmm6        # 16-byte Reload
	movaps	272(%rsp), %xmm1        # 16-byte Reload
	unpcklps	%xmm7, %xmm9    # xmm9 = xmm9[0],xmm7[0],xmm9[1],xmm7[1]
	shufps	$36, %xmm4, %xmm9       # xmm9 = xmm9[0,1],xmm4[2,0]
	addq	$6144, %r11             # imm = 0x1800
	addq	$4, %r12
	jne	.LBB5_7
# %bb.8:                                # %polly.loop_exit22
                                        #   in Loop: Header=BB5_6 Depth=5
	movups	%xmm8, (%r8)
	movaps	96(%rsp), %xmm0         # 16-byte Reload
	movups	%xmm0, 16(%r8)
	movups	%xmm6, 32(%r8)
	movups	%xmm1, 48(%r8)
	movaps	64(%rsp), %xmm0         # 16-byte Reload
	movups	%xmm0, 48(%rdx)
	movaps	48(%rsp), %xmm0         # 16-byte Reload
	movups	%xmm0, 32(%rdx)
	movaps	(%rsp), %xmm0           # 16-byte Reload
	movups	%xmm0, 16(%rdx)
	movups	%xmm15, (%rdx)
	movaps	80(%rsp), %xmm0         # 16-byte Reload
	movups	%xmm0, 48(%r13)
	movaps	112(%rsp), %xmm0        # 16-byte Reload
	movups	%xmm0, 16(%r13)
	movups	%xmm11, (%r13)
	movups	%xmm12, 32(%r13)
	movups	%xmm3, 48(%r10)
	movups	%xmm13, 16(%r10)
	movups	%xmm9, (%r10)
	movups	%xmm2, 32(%r10)
	addq	$6144, %r9              # imm = 0x1800
	cmpq	%rsi, %rax
	leaq	1(%rax), %rax
	jl	.LBB5_6
# %bb.9:                                # %polly.loop_exit16
                                        #   in Loop: Header=BB5_5 Depth=4
	movq	176(%rsp), %rax         # 8-byte Reload
	addq	$64, %rax
	addq	$393216, %r14           # imm = 0x60000
	movq	184(%rsp), %r9          # 8-byte Reload
	addq	$256, %r9               # imm = 0x100
	cmpq	$1536, %rax             # imm = 0x600
	movq	168(%rsp), %rdx         # 8-byte Reload
	jb	.LBB5_5
# %bb.10:                               # %polly.loop_exit10
                                        #   in Loop: Header=BB5_4 Depth=3
	movq	144(%rsp), %rax         # 8-byte Reload
	addq	$64, %rax
	movq	152(%rsp), %rcx         # 8-byte Reload
	incq	%rcx
	movq	160(%rsp), %r14         # 8-byte Reload
	addq	$256, %r14              # imm = 0x100
	cmpq	$1536, %rax             # imm = 0x600
	jb	.LBB5_4
# %bb.11:                               # %polly.loop_exit4
                                        #   in Loop: Header=BB5_3 Depth=2
	addq	$64, %rdx
	addq	$393216, 24(%rsp)       # 8-byte Folded Spill
                                        # imm = 0x60000
	cmpq	136(%rsp), %rdx         # 8-byte Folded Reload
	jle	.LBB5_3
.LBB5_1:                                # %polly.par.setup
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB5_3 Depth 2
                                        #       Child Loop BB5_4 Depth 3
                                        #         Child Loop BB5_5 Depth 4
                                        #           Child Loop BB5_6 Depth 5
                                        #             Child Loop BB5_7 Depth 6
	leaq	40(%rsp), %rdi
	leaq	32(%rsp), %rsi
	callq	GOMP_loop_runtime_next@PLT
	testb	%al, %al
	jne	.LBB5_2
# %bb.12:                               # %polly.par.exit
	callq	GOMP_loop_end_nowait@PLT
	addq	$296, %rsp              # imm = 0x128
	.cfi_def_cfa_offset 56
	popq	%rbx
	.cfi_def_cfa_offset 48
	popq	%r12
	.cfi_def_cfa_offset 40
	popq	%r13
	.cfi_def_cfa_offset 32
	popq	%r14
	.cfi_def_cfa_offset 24
	popq	%r15
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end5:
	.size	main_polly_subfn_1, .Lfunc_end5-main_polly_subfn_1
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
