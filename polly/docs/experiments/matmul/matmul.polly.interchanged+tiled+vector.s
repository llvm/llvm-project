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
.LBB0_1:                                # %polly.loop_header
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_2 Depth 2
	movl	$1, %edi
	xorl	%edx, %edx
	.p2align	4, 0x90
.LBB0_2:                                # %polly.loop_header1
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
# %bb.3:                                # %polly.loop_exit3
                                        #   in Loop: Header=BB0_1 Depth=1
	addq	$1, %r9
	addq	$6144, %rax             # imm = 0x1800
	addq	$6144, %rcx             # imm = 0x1800
	addl	$2, %r8d
	cmpq	$1536, %r9              # imm = 0x600
	jne	.LBB0_1
# %bb.4:                                # %polly.exiting
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
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$264, %rsp              # imm = 0x108
	.cfi_offset %rbx, -56
	.cfi_offset %r12, -48
	.cfi_offset %r13, -40
	.cfi_offset %r14, -32
	.cfi_offset %r15, -24
	callq	init_array
	leaq	C(%rip), %rdi
	xorl	%eax, %eax
	movq	%rax, -48(%rbp)         # 8-byte Spill
	xorl	%esi, %esi
	movl	$9437184, %edx          # imm = 0x900000
	callq	memset@PLT
	movl	$64, %eax
	movq	%rax, -80(%rbp)         # 8-byte Spill
	leaq	A(%rip), %rax
	movq	%rax, -72(%rbp)         # 8-byte Spill
	.p2align	4, 0x90
.LBB2_1:                                # %polly.loop_header8
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB2_2 Depth 2
                                        #       Child Loop BB2_3 Depth 3
                                        #         Child Loop BB2_4 Depth 4
                                        #           Child Loop BB2_5 Depth 5
	leaq	B+192(%rip), %r9
	xorl	%edi, %edi
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB2_2:                                # %polly.loop_header14
                                        #   Parent Loop BB2_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB2_3 Depth 3
                                        #         Child Loop BB2_4 Depth 4
                                        #           Child Loop BB2_5 Depth 5
	movq	%rax, -168(%rbp)        # 8-byte Spill
	movq	%rdi, -176(%rbp)        # 8-byte Spill
	shlq	$6, %rdi
	leaq	16(%rdi), %rdx
	leaq	32(%rdi), %rsi
	leaq	48(%rdi), %rcx
	movq	-72(%rbp), %r12         # 8-byte Reload
	movq	%r9, -184(%rbp)         # 8-byte Spill
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB2_3:                                # %polly.loop_header20
                                        #   Parent Loop BB2_1 Depth=1
                                        #     Parent Loop BB2_2 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB2_4 Depth 4
                                        #           Child Loop BB2_5 Depth 5
	movq	%rax, -192(%rbp)        # 8-byte Spill
	movq	%r12, -200(%rbp)        # 8-byte Spill
	movq	-48(%rbp), %r14         # 8-byte Reload
	.p2align	4, 0x90
.LBB2_4:                                # %polly.loop_header26
                                        #   Parent Loop BB2_1 Depth=1
                                        #     Parent Loop BB2_2 Depth=2
                                        #       Parent Loop BB2_3 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB2_5 Depth 5
	leaq	(%r14,%r14,2), %rbx
	shlq	$11, %rbx
	leaq	C(%rip), %rax
	addq	%rax, %rbx
	leaq	(%rbx,%rdi,4), %r8
	leaq	(%rbx,%rdx,4), %r15
	leaq	(%rbx,%rsi,4), %r10
	leaq	(%rbx,%rcx,4), %r11
	movups	(%rbx,%rdi,4), %xmm8
	movups	16(%rbx,%rdi,4), %xmm0
	movaps	%xmm0, -144(%rbp)       # 16-byte Spill
	movups	32(%rbx,%rdi,4), %xmm6
	movups	48(%rbx,%rdi,4), %xmm1
	movups	(%rbx,%rdx,4), %xmm15
	movups	16(%rbx,%rdx,4), %xmm0
	movaps	%xmm0, -64(%rbp)        # 16-byte Spill
	movups	32(%rbx,%rdx,4), %xmm0
	movaps	%xmm0, -96(%rbp)        # 16-byte Spill
	movups	48(%rbx,%rdx,4), %xmm0
	movaps	%xmm0, -112(%rbp)       # 16-byte Spill
	movups	(%rbx,%rsi,4), %xmm11
	movups	16(%rbx,%rsi,4), %xmm0
	movaps	%xmm0, -160(%rbp)       # 16-byte Spill
	movups	32(%rbx,%rsi,4), %xmm12
	movups	48(%rbx,%rsi,4), %xmm0
	movaps	%xmm0, -128(%rbp)       # 16-byte Spill
	movups	(%rbx,%rcx,4), %xmm9
	movups	16(%rbx,%rcx,4), %xmm13
	movups	32(%rbx,%rcx,4), %xmm2
	movups	48(%rbx,%rcx,4), %xmm3
	movq	%r9, %rbx
	movl	$0, %r13d
	.p2align	4, 0x90
.LBB2_5:                                # %vector.ph
                                        #   Parent Loop BB2_1 Depth=1
                                        #     Parent Loop BB2_2 Depth=2
                                        #       Parent Loop BB2_3 Depth=3
                                        #         Parent Loop BB2_4 Depth=4
                                        # =>        This Inner Loop Header: Depth=5
	movaps	%xmm12, -240(%rbp)      # 16-byte Spill
	movaps	%xmm2, -256(%rbp)       # 16-byte Spill
	movaps	%xmm3, -272(%rbp)       # 16-byte Spill
	movaps	%xmm8, %xmm10
	movaps	-144(%rbp), %xmm7       # 16-byte Reload
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
	movaps	-160(%rbx), %xmm0
	movaps	-144(%rbx), %xmm1
	movaps	%xmm1, %xmm6
	shufps	$0, %xmm0, %xmm6        # xmm6 = xmm6[0,0],xmm0[0,0]
	movaps	-192(%rbx), %xmm3
	movaps	-176(%rbx), %xmm4
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
	movss	(%r12,%r13,4), %xmm0    # xmm0 = mem[0],zero,zero,zero
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
	movaps	%xmm1, -304(%rbp)       # 16-byte Spill
	movaps	%xmm4, %xmm0
	shufps	$34, %xmm14, %xmm0      # xmm0 = xmm0[2,0],xmm14[2,0]
	movaps	%xmm8, %xmm1
	unpckhps	%xmm6, %xmm1    # xmm1 = xmm1[2],xmm6[2],xmm1[3],xmm6[3]
	shufps	$36, %xmm0, %xmm1       # xmm1 = xmm1[0,1],xmm0[2,0]
	movaps	%xmm1, -288(%rbp)       # 16-byte Spill
	movaps	%xmm14, %xmm0
	unpcklps	%xmm4, %xmm0    # xmm0 = xmm0[0],xmm4[0],xmm0[1],xmm4[1]
	movaps	%xmm6, %xmm1
	shufps	$17, %xmm8, %xmm1       # xmm1 = xmm1[1,0],xmm8[1,0]
	shufps	$226, %xmm0, %xmm1      # xmm1 = xmm1[2,0],xmm0[2,3]
	movaps	%xmm1, -144(%rbp)       # 16-byte Spill
	shufps	$0, %xmm14, %xmm4       # xmm4 = xmm4[0,0],xmm14[0,0]
	unpcklps	%xmm6, %xmm8    # xmm8 = xmm8[0],xmm6[0],xmm8[1],xmm6[1]
	shufps	$36, %xmm4, %xmm8       # xmm8 = xmm8[0,1],xmm4[2,0]
	movaps	%xmm15, %xmm14
	movaps	-64(%rbp), %xmm4        # 16-byte Reload
	unpcklps	%xmm4, %xmm14   # xmm14 = xmm14[0],xmm4[0],xmm14[1],xmm4[1]
	movaps	-112(%rbp), %xmm1       # 16-byte Reload
	movaps	%xmm1, %xmm0
	movaps	-96(%rbp), %xmm3        # 16-byte Reload
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
	movaps	%xmm4, -64(%rbp)        # 16-byte Spill
	movaps	-96(%rbx), %xmm2
	movaps	-80(%rbx), %xmm1
	movaps	%xmm1, %xmm4
	shufps	$0, %xmm2, %xmm4        # xmm4 = xmm4[0,0],xmm2[0,0]
	movaps	-112(%rbx), %xmm10
	movaps	-128(%rbx), %xmm0
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
	movaps	%xmm5, -224(%rbp)       # 16-byte Spill
	mulps	%xmm5, %xmm15
	addps	%xmm14, %xmm15
	mulps	%xmm5, %xmm6
	addps	%xmm12, %xmm6
	mulps	%xmm5, %xmm4
	addps	%xmm7, %xmm4
	mulps	%xmm5, %xmm10
	addps	-64(%rbp), %xmm10       # 16-byte Folded Reload
	movaps	%xmm4, %xmm0
	unpckhps	%xmm10, %xmm0   # xmm0 = xmm0[2],xmm10[2],xmm0[3],xmm10[3]
	movaps	%xmm6, %xmm1
	shufps	$51, %xmm15, %xmm1      # xmm1 = xmm1[3,0],xmm15[3,0]
	shufps	$226, %xmm0, %xmm1      # xmm1 = xmm1[2,0],xmm0[2,3]
	movaps	%xmm1, -112(%rbp)       # 16-byte Spill
	movaps	%xmm10, %xmm0
	shufps	$34, %xmm4, %xmm0       # xmm0 = xmm0[2,0],xmm4[2,0]
	movaps	%xmm15, %xmm1
	unpckhps	%xmm6, %xmm1    # xmm1 = xmm1[2],xmm6[2],xmm1[3],xmm6[3]
	shufps	$36, %xmm0, %xmm1       # xmm1 = xmm1[0,1],xmm0[2,0]
	movaps	%xmm1, -96(%rbp)        # 16-byte Spill
	movaps	%xmm4, %xmm0
	unpcklps	%xmm10, %xmm0   # xmm0 = xmm0[0],xmm10[0],xmm0[1],xmm10[1]
	movaps	%xmm6, %xmm1
	shufps	$17, %xmm15, %xmm1      # xmm1 = xmm1[1,0],xmm15[1,0]
	shufps	$226, %xmm0, %xmm1      # xmm1 = xmm1[2,0],xmm0[2,3]
	movaps	%xmm1, -64(%rbp)        # 16-byte Spill
	shufps	$0, %xmm4, %xmm10       # xmm10 = xmm10[0,0],xmm4[0,0]
	unpcklps	%xmm6, %xmm15   # xmm15 = xmm15[0],xmm6[0],xmm15[1],xmm6[1]
	shufps	$36, %xmm10, %xmm15     # xmm15 = xmm15[0,1],xmm10[2,0]
	movaps	%xmm11, %xmm10
	movaps	-160(%rbp), %xmm14      # 16-byte Reload
	unpcklps	%xmm14, %xmm10  # xmm10 = xmm10[0],xmm14[0],xmm10[1],xmm14[1]
	movaps	-128(%rbp), %xmm2       # 16-byte Reload
	movaps	%xmm2, %xmm0
	movaps	-240(%rbp), %xmm3       # 16-byte Reload
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
	movaps	-32(%rbx), %xmm1
	movaps	-16(%rbx), %xmm2
	movaps	%xmm2, %xmm3
	shufps	$0, %xmm1, %xmm3        # xmm3 = xmm3[0,0],xmm1[0,0]
	movaps	-48(%rbx), %xmm4
	movaps	-64(%rbx), %xmm5
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
	movaps	-224(%rbp), %xmm1       # 16-byte Reload
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
	movaps	%xmm1, -128(%rbp)       # 16-byte Spill
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
	movaps	%xmm1, -160(%rbp)       # 16-byte Spill
	shufps	$0, %xmm6, %xmm4        # xmm4 = xmm4[0,0],xmm6[0,0]
	unpcklps	%xmm7, %xmm11   # xmm11 = xmm11[0],xmm7[0],xmm11[1],xmm7[1]
	shufps	$36, %xmm4, %xmm11      # xmm11 = xmm11[0,1],xmm4[2,0]
	movaps	%xmm9, %xmm10
	unpcklps	%xmm13, %xmm10  # xmm10 = xmm10[0],xmm13[0],xmm10[1],xmm13[1]
	movaps	-272(%rbp), %xmm2       # 16-byte Reload
	movaps	%xmm2, %xmm0
	movaps	-256(%rbp), %xmm3       # 16-byte Reload
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
	movaps	32(%rbx), %xmm1
	movaps	48(%rbx), %xmm2
	movaps	%xmm2, %xmm3
	shufps	$0, %xmm1, %xmm3        # xmm3 = xmm3[0,0],xmm1[0,0]
	movaps	16(%rbx), %xmm4
	movaps	(%rbx), %xmm5
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
	movaps	-224(%rbp), %xmm1       # 16-byte Reload
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
	movaps	-288(%rbp), %xmm6       # 16-byte Reload
	movaps	-304(%rbp), %xmm1       # 16-byte Reload
	unpcklps	%xmm7, %xmm9    # xmm9 = xmm9[0],xmm7[0],xmm9[1],xmm7[1]
	shufps	$36, %xmm4, %xmm9       # xmm9 = xmm9[0,1],xmm4[2,0]
	addq	$1, %r13
	addq	$6144, %rbx             # imm = 0x1800
	cmpq	$64, %r13
	jne	.LBB2_5
# %bb.6:                                # %polly.loop_exit34
                                        #   in Loop: Header=BB2_4 Depth=4
	movups	%xmm8, (%r8)
	movaps	-144(%rbp), %xmm0       # 16-byte Reload
	movups	%xmm0, 16(%r8)
	movups	%xmm6, 32(%r8)
	movups	%xmm1, 48(%r8)
	movaps	-112(%rbp), %xmm0       # 16-byte Reload
	movups	%xmm0, 48(%r15)
	movaps	-96(%rbp), %xmm0        # 16-byte Reload
	movups	%xmm0, 32(%r15)
	movaps	-64(%rbp), %xmm0        # 16-byte Reload
	movups	%xmm0, 16(%r15)
	movups	%xmm15, (%r15)
	movaps	-128(%rbp), %xmm0       # 16-byte Reload
	movups	%xmm0, 48(%r10)
	movaps	-160(%rbp), %xmm0       # 16-byte Reload
	movups	%xmm0, 16(%r10)
	movups	%xmm11, (%r10)
	movups	%xmm12, 32(%r10)
	movups	%xmm3, 48(%r11)
	movups	%xmm13, 16(%r11)
	movups	%xmm9, (%r11)
	movups	%xmm2, 32(%r11)
	addq	$1, %r14
	addq	$6144, %r12             # imm = 0x1800
	cmpq	-80(%rbp), %r14         # 8-byte Folded Reload
	jne	.LBB2_4
# %bb.7:                                # %polly.loop_exit28
                                        #   in Loop: Header=BB2_3 Depth=3
	movq	-192(%rbp), %rax        # 8-byte Reload
	addq	$64, %rax
	addq	$393216, %r9            # imm = 0x60000
	movq	-200(%rbp), %r12        # 8-byte Reload
	addq	$256, %r12              # imm = 0x100
	cmpq	$1536, %rax             # imm = 0x600
	jb	.LBB2_3
# %bb.8:                                # %polly.loop_exit22
                                        #   in Loop: Header=BB2_2 Depth=2
	movq	-168(%rbp), %rax        # 8-byte Reload
	addq	$64, %rax
	movq	-176(%rbp), %rdi        # 8-byte Reload
	addq	$1, %rdi
	movq	-184(%rbp), %r9         # 8-byte Reload
	addq	$256, %r9               # imm = 0x100
	cmpq	$1536, %rax             # imm = 0x600
	jb	.LBB2_2
# %bb.9:                                # %polly.loop_exit16
                                        #   in Loop: Header=BB2_1 Depth=1
	movq	-48(%rbp), %rax         # 8-byte Reload
	movq	%rax, %rcx
	addq	$64, %rcx
	addq	$64, -80(%rbp)          # 8-byte Folded Spill
	addq	$393216, -72(%rbp)      # 8-byte Folded Spill
                                        # imm = 0x60000
	movq	%rcx, %rax
	movq	%rcx, -48(%rbp)         # 8-byte Spill
	cmpq	$1536, %rcx             # imm = 0x600
	jb	.LBB2_1
# %bb.10:                               # %polly.exiting
	xorl	%eax, %eax
	addq	$264, %rsp              # imm = 0x108
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
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
