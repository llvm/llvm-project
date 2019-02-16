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
	subq	$344, %rsp              # imm = 0x158
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
	movq	%rax, -64(%rbp)         # 8-byte Spill
	leaq	A(%rip), %rax
	movq	%rax, -56(%rbp)         # 8-byte Spill
	.p2align	4, 0x90
.LBB2_1:                                # %polly.loop_header8
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB2_2 Depth 2
                                        #       Child Loop BB2_3 Depth 3
                                        #         Child Loop BB2_4 Depth 4
                                        #           Child Loop BB2_5 Depth 5
	leaq	B+240(%rip), %rax
	xorl	%edi, %edi
	.p2align	4, 0x90
.LBB2_2:                                # %polly.loop_header14
                                        #   Parent Loop BB2_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB2_3 Depth 3
                                        #         Child Loop BB2_4 Depth 4
                                        #           Child Loop BB2_5 Depth 5
	movq	%rdi, %rcx
	orq	$4, %rcx
	movq	%rcx, -80(%rbp)         # 8-byte Spill
	movq	%rdi, %rcx
	orq	$8, %rcx
	movq	%rcx, -264(%rbp)        # 8-byte Spill
	movq	%rdi, %rcx
	orq	$12, %rcx
	movq	%rcx, -256(%rbp)        # 8-byte Spill
	movq	%rdi, %rcx
	orq	$16, %rcx
	movq	%rcx, -248(%rbp)        # 8-byte Spill
	movq	%rdi, %rcx
	orq	$20, %rcx
	movq	%rcx, -240(%rbp)        # 8-byte Spill
	movq	%rdi, %rcx
	orq	$24, %rcx
	movq	%rcx, -232(%rbp)        # 8-byte Spill
	movq	%rdi, %rcx
	orq	$28, %rcx
	movq	%rcx, -224(%rbp)        # 8-byte Spill
	movq	%rdi, %rcx
	orq	$32, %rcx
	movq	%rcx, -216(%rbp)        # 8-byte Spill
	movq	%rdi, %rcx
	orq	$36, %rcx
	movq	%rcx, -208(%rbp)        # 8-byte Spill
	movq	%rdi, %rcx
	orq	$40, %rcx
	movq	%rcx, -200(%rbp)        # 8-byte Spill
	movq	%rdi, %rcx
	orq	$44, %rcx
	movq	%rcx, -192(%rbp)        # 8-byte Spill
	movq	%rdi, %rcx
	orq	$48, %rcx
	movq	%rcx, -184(%rbp)        # 8-byte Spill
	movq	%rdi, %rcx
	orq	$52, %rcx
	movq	%rcx, -176(%rbp)        # 8-byte Spill
	movq	%rdi, %rcx
	orq	$56, %rcx
	movq	%rcx, -168(%rbp)        # 8-byte Spill
	movq	%rdi, %rcx
	orq	$60, %rcx
	movq	%rcx, -160(%rbp)        # 8-byte Spill
	movq	-56(%rbp), %rdx         # 8-byte Reload
	movq	%rax, -136(%rbp)        # 8-byte Spill
	movq	%rax, -72(%rbp)         # 8-byte Spill
	xorl	%eax, %eax
	movq	%rdi, -272(%rbp)        # 8-byte Spill
	.p2align	4, 0x90
.LBB2_3:                                # %polly.loop_header20
                                        #   Parent Loop BB2_1 Depth=1
                                        #     Parent Loop BB2_2 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB2_4 Depth 4
                                        #           Child Loop BB2_5 Depth 5
	movq	%rax, -144(%rbp)        # 8-byte Spill
	movq	%rdx, -152(%rbp)        # 8-byte Spill
	movq	-48(%rbp), %rax         # 8-byte Reload
	.p2align	4, 0x90
.LBB2_4:                                # %polly.loop_header26
                                        #   Parent Loop BB2_1 Depth=1
                                        #     Parent Loop BB2_2 Depth=2
                                        #       Parent Loop BB2_3 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB2_5 Depth 5
	movq	%rax, -376(%rbp)        # 8-byte Spill
	leaq	(%rax,%rax,2), %rax
	shlq	$11, %rax
	leaq	C(%rip), %rsi
	addq	%rsi, %rax
	leaq	(%rax,%rdi,4), %rcx
	movq	%rcx, -368(%rbp)        # 8-byte Spill
	movq	-80(%rbp), %rcx         # 8-byte Reload
	leaq	(%rax,%rcx,4), %rcx
	movq	%rcx, -360(%rbp)        # 8-byte Spill
	movq	-264(%rbp), %rbx        # 8-byte Reload
	leaq	(%rax,%rbx,4), %rcx
	movq	%rcx, -352(%rbp)        # 8-byte Spill
	movq	-256(%rbp), %r8         # 8-byte Reload
	movq	%rdi, %rsi
	leaq	(%rax,%r8,4), %rdi
	movq	%rdi, -344(%rbp)        # 8-byte Spill
	movq	-248(%rbp), %rdi        # 8-byte Reload
	leaq	(%rax,%rdi,4), %rcx
	movq	%rcx, -336(%rbp)        # 8-byte Spill
	movq	-240(%rbp), %r9         # 8-byte Reload
	leaq	(%rax,%r9,4), %rcx
	movq	%rcx, -328(%rbp)        # 8-byte Spill
	movq	-232(%rbp), %r10        # 8-byte Reload
	leaq	(%rax,%r10,4), %rcx
	movq	%rcx, -320(%rbp)        # 8-byte Spill
	movq	-224(%rbp), %r14        # 8-byte Reload
	leaq	(%rax,%r14,4), %rcx
	movq	%rcx, -312(%rbp)        # 8-byte Spill
	movq	-216(%rbp), %r15        # 8-byte Reload
	leaq	(%rax,%r15,4), %rcx
	movq	%rcx, -304(%rbp)        # 8-byte Spill
	movq	-208(%rbp), %r12        # 8-byte Reload
	leaq	(%rax,%r12,4), %rcx
	movq	%rcx, -296(%rbp)        # 8-byte Spill
	movq	-200(%rbp), %r13        # 8-byte Reload
	leaq	(%rax,%r13,4), %rcx
	movq	%rcx, -288(%rbp)        # 8-byte Spill
	movq	-192(%rbp), %r11        # 8-byte Reload
	leaq	(%rax,%r11,4), %rcx
	movq	%rcx, -280(%rbp)        # 8-byte Spill
	movaps	(%rax,%rsi,4), %xmm15
	movq	-80(%rbp), %rcx         # 8-byte Reload
	movaps	(%rax,%rcx,4), %xmm14
	movaps	(%rax,%rbx,4), %xmm13
	movaps	(%rax,%r8,4), %xmm12
	movaps	(%rax,%rdi,4), %xmm11
	movaps	(%rax,%r9,4), %xmm10
	movaps	(%rax,%r10,4), %xmm9
	movaps	(%rax,%r14,4), %xmm8
	movaps	(%rax,%r15,4), %xmm7
	movaps	(%rax,%r12,4), %xmm6
	movaps	(%rax,%r13,4), %xmm5
	movaps	(%rax,%r11,4), %xmm4
	movq	-184(%rbp), %rcx        # 8-byte Reload
	movaps	(%rax,%rcx,4), %xmm3
	movq	-176(%rbp), %rsi        # 8-byte Reload
	movaps	(%rax,%rsi,4), %xmm0
	movaps	%xmm0, -96(%rbp)        # 16-byte Spill
	movq	-168(%rbp), %rbx        # 8-byte Reload
	movaps	(%rax,%rbx,4), %xmm0
	movaps	%xmm0, -112(%rbp)       # 16-byte Spill
	movq	-160(%rbp), %rdi        # 8-byte Reload
	movaps	(%rax,%rdi,4), %xmm0
	movaps	%xmm0, -128(%rbp)       # 16-byte Spill
	leaq	(%rax,%rcx,4), %r8
	leaq	(%rax,%rsi,4), %rcx
	leaq	(%rax,%rbx,4), %rsi
	leaq	(%rax,%rdi,4), %rax
	movq	-72(%rbp), %r9          # 8-byte Reload
	movl	$0, %r10d
	.p2align	4, 0x90
.LBB2_5:                                # %vector.ph
                                        #   Parent Loop BB2_1 Depth=1
                                        #     Parent Loop BB2_2 Depth=2
                                        #       Parent Loop BB2_3 Depth=3
                                        #         Parent Loop BB2_4 Depth=4
                                        # =>        This Inner Loop Header: Depth=5
	movss	(%rdx,%r10,4), %xmm0    # xmm0 = mem[0],zero,zero,zero
	shufps	$0, %xmm0, %xmm0        # xmm0 = xmm0[0,0,0,0]
	movaps	-240(%r9), %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm15
	movaps	-224(%r9), %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm14
	movaps	-208(%r9), %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm13
	movaps	-192(%r9), %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm12
	movaps	-176(%r9), %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm11
	movaps	-160(%r9), %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm10
	movaps	-144(%r9), %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm9
	movaps	-128(%r9), %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm8
	movaps	-112(%r9), %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm7
	movaps	-96(%r9), %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm6
	movaps	-80(%r9), %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm5
	movaps	-64(%r9), %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm4
	movaps	-48(%r9), %xmm1
	mulps	%xmm0, %xmm1
	addps	%xmm1, %xmm3
	movaps	-32(%r9), %xmm1
	mulps	%xmm0, %xmm1
	movaps	-96(%rbp), %xmm2        # 16-byte Reload
	addps	%xmm1, %xmm2
	movaps	%xmm2, -96(%rbp)        # 16-byte Spill
	movaps	-16(%r9), %xmm1
	mulps	%xmm0, %xmm1
	movaps	-112(%rbp), %xmm2       # 16-byte Reload
	addps	%xmm1, %xmm2
	movaps	%xmm2, -112(%rbp)       # 16-byte Spill
	mulps	(%r9), %xmm0
	movaps	-128(%rbp), %xmm1       # 16-byte Reload
	addps	%xmm0, %xmm1
	movaps	%xmm1, -128(%rbp)       # 16-byte Spill
	addq	$1, %r10
	addq	$6144, %r9              # imm = 0x1800
	cmpq	$64, %r10
	jne	.LBB2_5
# %bb.6:                                # %polly.loop_exit34
                                        #   in Loop: Header=BB2_4 Depth=4
	movq	-368(%rbp), %rdi        # 8-byte Reload
	movaps	%xmm15, (%rdi)
	movq	-360(%rbp), %rdi        # 8-byte Reload
	movaps	%xmm14, (%rdi)
	movq	-352(%rbp), %rdi        # 8-byte Reload
	movaps	%xmm13, (%rdi)
	movq	-344(%rbp), %rdi        # 8-byte Reload
	movaps	%xmm12, (%rdi)
	movq	-336(%rbp), %rdi        # 8-byte Reload
	movaps	%xmm11, (%rdi)
	movq	-328(%rbp), %rdi        # 8-byte Reload
	movaps	%xmm10, (%rdi)
	movq	-320(%rbp), %rdi        # 8-byte Reload
	movaps	%xmm9, (%rdi)
	movq	-312(%rbp), %rdi        # 8-byte Reload
	movaps	%xmm8, (%rdi)
	movq	-304(%rbp), %rdi        # 8-byte Reload
	movaps	%xmm7, (%rdi)
	movq	-296(%rbp), %rdi        # 8-byte Reload
	movaps	%xmm6, (%rdi)
	movq	-288(%rbp), %rdi        # 8-byte Reload
	movaps	%xmm5, (%rdi)
	movq	-280(%rbp), %rdi        # 8-byte Reload
	movaps	%xmm4, (%rdi)
	movaps	%xmm3, (%r8)
	movaps	-96(%rbp), %xmm0        # 16-byte Reload
	movaps	%xmm0, (%rcx)
	movaps	-112(%rbp), %xmm0       # 16-byte Reload
	movaps	%xmm0, (%rsi)
	movaps	-128(%rbp), %xmm0       # 16-byte Reload
	movaps	%xmm0, (%rax)
	movq	-376(%rbp), %rax        # 8-byte Reload
	addq	$1, %rax
	addq	$6144, %rdx             # imm = 0x1800
	cmpq	-64(%rbp), %rax         # 8-byte Folded Reload
	movq	-272(%rbp), %rdi        # 8-byte Reload
	jne	.LBB2_4
# %bb.7:                                # %polly.loop_exit28
                                        #   in Loop: Header=BB2_3 Depth=3
	movq	-144(%rbp), %rax        # 8-byte Reload
	addq	$64, %rax
	addq	$393216, -72(%rbp)      # 8-byte Folded Spill
                                        # imm = 0x60000
	movq	-152(%rbp), %rdx        # 8-byte Reload
	addq	$256, %rdx              # imm = 0x100
	cmpq	$1536, %rax             # imm = 0x600
	jb	.LBB2_3
# %bb.8:                                # %polly.loop_exit22
                                        #   in Loop: Header=BB2_2 Depth=2
	addq	$64, %rdi
	movq	-136(%rbp), %rax        # 8-byte Reload
	addq	$256, %rax              # imm = 0x100
	cmpq	$1536, %rdi             # imm = 0x600
	jb	.LBB2_2
# %bb.9:                                # %polly.loop_exit16
                                        #   in Loop: Header=BB2_1 Depth=1
	movq	-48(%rbp), %rax         # 8-byte Reload
	movq	%rax, %rcx
	addq	$64, %rcx
	addq	$64, -64(%rbp)          # 8-byte Folded Spill
	addq	$393216, -56(%rbp)      # 8-byte Folded Spill
                                        # imm = 0x60000
	movq	%rcx, %rax
	movq	%rcx, -48(%rbp)         # 8-byte Spill
	cmpq	$1536, %rcx             # imm = 0x600
	jb	.LBB2_1
# %bb.10:                               # %polly.exiting
	xorl	%eax, %eax
	addq	$344, %rsp              # imm = 0x158
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
