	.file	"matmul.polly.interchanged+tiled+vector.ll"
	.section	.rodata.cst8,"aM",@progbits,8
	.align	8
.LCPI0_0:
	.quad	4602678819172646912     # double 5.000000e-01
	.text
	.globl	init_array
	.align	16, 0x90
	.type	init_array,@function
init_array:                             # @init_array
# BB#0:                                 # %pollyBB
	xorl	%eax, %eax
	movsd	.LCPI0_0(%rip), %xmm0
	movq	%rax, %rcx
	.align	16, 0x90
.LBB0_2:                                # %polly.loop_header1.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_3 Depth 2
	movq	$-1536, %rdx            # imm = 0xFFFFFFFFFFFFFA00
	xorl	%esi, %esi
	.align	16, 0x90
.LBB0_3:                                # %polly.loop_body2
                                        #   Parent Loop BB0_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%esi, %edi
	sarl	$31, %edi
	shrl	$22, %edi
	addl	%esi, %edi
	andl	$-1024, %edi            # imm = 0xFFFFFFFFFFFFFC00
	negl	%edi
	leal	1(%rsi,%rdi), %edi
	cvtsi2sd	%edi, %xmm1
	mulsd	%xmm0, %xmm1
	cvtsd2ss	%xmm1, %xmm1
	movss	%xmm1, A+6144(%rax,%rdx,4)
	movss	%xmm1, B+6144(%rax,%rdx,4)
	addl	%ecx, %esi
	incq	%rdx
	jne	.LBB0_3
# BB#1:                                 # %polly.loop_header.loopexit
                                        #   in Loop: Header=BB0_2 Depth=1
	addq	$6144, %rax             # imm = 0x1800
	incq	%rcx
	cmpq	$1536, %rcx             # imm = 0x600
	jne	.LBB0_2
# BB#4:                                 # %polly.after_loop
	ret
.Ltmp0:
	.size	init_array, .Ltmp0-init_array

	.globl	print_array
	.align	16, 0x90
	.type	print_array,@function
print_array:                            # @print_array
# BB#0:
	pushq	%r14
	pushq	%rbx
	pushq	%rax
	movq	$-9437184, %rbx         # imm = 0xFFFFFFFFFF700000
	.align	16, 0x90
.LBB1_1:                                # %.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_2 Depth 2
	xorl	%r14d, %r14d
	movq	stdout(%rip), %rdi
	.align	16, 0x90
.LBB1_2:                                #   Parent Loop BB1_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	C+9437184(%rbx,%r14,4), %xmm0
	cvtss2sd	%xmm0, %xmm0
	movl	$.L.str, %esi
	movb	$1, %al
	callq	fprintf
	movslq	%r14d, %rax
	imulq	$1717986919, %rax, %rcx # imm = 0x66666667
	movq	%rcx, %rdx
	shrq	$63, %rdx
	sarq	$37, %rcx
	addl	%edx, %ecx
	imull	$80, %ecx, %ecx
	subl	%ecx, %eax
	cmpl	$79, %eax
	jne	.LBB1_4
# BB#3:                                 #   in Loop: Header=BB1_2 Depth=2
	movq	stdout(%rip), %rsi
	movl	$10, %edi
	callq	fputc
.LBB1_4:                                #   in Loop: Header=BB1_2 Depth=2
	incq	%r14
	movq	stdout(%rip), %rsi
	cmpq	$1536, %r14             # imm = 0x600
	movq	%rsi, %rdi
	jne	.LBB1_2
# BB#5:                                 #   in Loop: Header=BB1_1 Depth=1
	movl	$10, %edi
	callq	fputc
	addq	$6144, %rbx             # imm = 0x1800
	jne	.LBB1_1
# BB#6:
	addq	$8, %rsp
	popq	%rbx
	popq	%r14
	ret
.Ltmp1:
	.size	print_array, .Ltmp1-print_array

	.section	.rodata.cst8,"aM",@progbits,8
	.align	8
.LCPI2_0:
	.quad	4602678819172646912     # double 5.000000e-01
	.text
	.globl	main
	.align	16, 0x90
	.type	main,@function
main:                                   # @main
# BB#0:                                 # %pollyBB
	pushq	%rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$24, %rsp
	xorl	%eax, %eax
	movsd	.LCPI2_0(%rip), %xmm0
	movq	%rax, %rcx
	.align	16, 0x90
.LBB2_1:                                # %polly.loop_header1.preheader.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB2_2 Depth 2
	movq	$-1536, %rdx            # imm = 0xFFFFFFFFFFFFFA00
	xorl	%esi, %esi
	.align	16, 0x90
.LBB2_2:                                # %polly.loop_body2.i
                                        #   Parent Loop BB2_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movl	%esi, %edi
	sarl	$31, %edi
	shrl	$22, %edi
	addl	%esi, %edi
	andl	$-1024, %edi            # imm = 0xFFFFFFFFFFFFFC00
	negl	%edi
	leal	1(%rsi,%rdi), %edi
	cvtsi2sd	%edi, %xmm1
	mulsd	%xmm0, %xmm1
	cvtsd2ss	%xmm1, %xmm1
	movss	%xmm1, A+6144(%rax,%rdx,4)
	movss	%xmm1, B+6144(%rax,%rdx,4)
	addl	%ecx, %esi
	incq	%rdx
	jne	.LBB2_2
# BB#3:                                 # %polly.loop_header.loopexit.i
                                        #   in Loop: Header=BB2_1 Depth=1
	addq	$6144, %rax             # imm = 0x1800
	incq	%rcx
	cmpq	$1536, %rcx             # imm = 0x600
	jne	.LBB2_1
# BB#4:                                 # %polly.loop_header.preheader
	movl	$C, %edi
	xorl	%esi, %esi
	movl	$9437184, %edx          # imm = 0x900000
	callq	memset
	xorl	%eax, %eax
	movq	%rax, 16(%rsp)          # 8-byte Spill
	movq	%rax, (%rsp)            # 8-byte Spill
	jmp	.LBB2_6
	.align	16, 0x90
.LBB2_5:                                # %polly.loop_header7.loopexit
                                        #   in Loop: Header=BB2_6 Depth=1
	addq	$393216, (%rsp)         # 8-byte Folded Spill
                                        # imm = 0x60000
	movq	16(%rsp), %rax          # 8-byte Reload
	addq	$64, %rax
	movq	%rax, 16(%rsp)          # 8-byte Spill
	cmpq	$1536, %rax             # imm = 0x600
	je	.LBB2_7
.LBB2_6:                                # %polly.loop_header12.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB2_9 Depth 2
                                        #       Child Loop BB2_11 Depth 3
                                        #         Child Loop BB2_14 Depth 4
                                        #           Child Loop BB2_18 Depth 5
                                        #             Child Loop BB2_19 Depth 6
	movq	16(%rsp), %rax          # 8-byte Reload
	leaq	63(%rax), %rax
	movq	(%rsp), %rcx            # 8-byte Reload
	leaq	A(%rcx), %rdx
	movq	%rdx, 8(%rsp)           # 8-byte Spill
	xorl	%edx, %edx
	jmp	.LBB2_9
	.align	16, 0x90
.LBB2_8:                                # %polly.loop_header12.loopexit
                                        #   in Loop: Header=BB2_9 Depth=2
	addq	$256, %rcx              # imm = 0x100
	addq	$64, %rdx
	cmpq	$1536, %rdx             # imm = 0x600
	je	.LBB2_5
.LBB2_9:                                # %polly.loop_header17.preheader
                                        #   Parent Loop BB2_6 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB2_11 Depth 3
                                        #         Child Loop BB2_14 Depth 4
                                        #           Child Loop BB2_18 Depth 5
                                        #             Child Loop BB2_19 Depth 6
	leaq	63(%rdx), %rsi
	xorl	%edi, %edi
	movq	8(%rsp), %r8            # 8-byte Reload
	movq	%rdx, %r9
	jmp	.LBB2_11
	.align	16, 0x90
.LBB2_10:                               # %polly.loop_header17.loopexit
                                        #   in Loop: Header=BB2_11 Depth=3
	addq	$256, %r8               # imm = 0x100
	addq	$98304, %r9             # imm = 0x18000
	addq	$64, %rdi
	cmpq	$1536, %rdi             # imm = 0x600
	je	.LBB2_8
.LBB2_11:                               # %polly.loop_body18
                                        #   Parent Loop BB2_6 Depth=1
                                        #     Parent Loop BB2_9 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB2_14 Depth 4
                                        #           Child Loop BB2_18 Depth 5
                                        #             Child Loop BB2_19 Depth 6
	cmpq	%rax, 16(%rsp)          # 8-byte Folded Reload
	jg	.LBB2_10
# BB#12:                                # %polly.loop_body23.lr.ph
                                        #   in Loop: Header=BB2_11 Depth=3
	leaq	63(%rdi), %r10
	xorl	%r11d, %r11d
	jmp	.LBB2_14
	.align	16, 0x90
.LBB2_13:                               # %polly.loop_header22.loopexit
                                        #   in Loop: Header=BB2_14 Depth=4
	addq	$6144, %r11             # imm = 0x1800
	cmpq	$393216, %r11           # imm = 0x60000
	je	.LBB2_10
.LBB2_14:                               # %polly.loop_body23
                                        #   Parent Loop BB2_6 Depth=1
                                        #     Parent Loop BB2_9 Depth=2
                                        #       Parent Loop BB2_11 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB2_18 Depth 5
                                        #             Child Loop BB2_19 Depth 6
	cmpq	%r10, %rdi
	jg	.LBB2_13
# BB#15:                                # %polly.loop_body23
                                        #   in Loop: Header=BB2_14 Depth=4
	cmpq	%rsi, %rdx
	jg	.LBB2_13
# BB#16:                                # %polly.loop_body33.lr.ph.preheader
                                        #   in Loop: Header=BB2_14 Depth=4
	leaq	(%r8,%r11), %rbx
	xorl	%r14d, %r14d
	movq	%r9, %r15
	movq	%r14, %r12
	jmp	.LBB2_18
	.align	16, 0x90
.LBB2_17:                               # %polly.loop_header27.loopexit
                                        #   in Loop: Header=BB2_18 Depth=5
	addq	$1536, %r15             # imm = 0x600
	incq	%r12
	cmpq	$64, %r12
	je	.LBB2_13
.LBB2_18:                               # %polly.loop_body33.lr.ph
                                        #   Parent Loop BB2_6 Depth=1
                                        #     Parent Loop BB2_9 Depth=2
                                        #       Parent Loop BB2_11 Depth=3
                                        #         Parent Loop BB2_14 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB2_19 Depth 6
	movss	(%rbx,%r12,4), %xmm0
	pshufd	$0, %xmm0, %xmm0        # xmm0 = xmm0[0,0,0,0]
	movq	%r14, %r13
	.align	16, 0x90
.LBB2_19:                               # %polly.loop_body33
                                        #   Parent Loop BB2_6 Depth=1
                                        #     Parent Loop BB2_9 Depth=2
                                        #       Parent Loop BB2_11 Depth=3
                                        #         Parent Loop BB2_14 Depth=4
                                        #           Parent Loop BB2_18 Depth=5
                                        # =>          This Inner Loop Header: Depth=6
	movaps	B(%r13,%r15,4), %xmm1
	mulps	%xmm0, %xmm1
	leaq	(%r11,%r13), %rbp
	addps	C(%rcx,%rbp), %xmm1
	movaps	%xmm1, C(%rcx,%rbp)
	addq	$16, %r13
	cmpq	$256, %r13              # imm = 0x100
	jne	.LBB2_19
	jmp	.LBB2_17
.LBB2_7:                                # %polly.after_loop9
	xorl	%eax, %eax
	addq	$24, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	ret
.Ltmp2:
	.size	main, .Ltmp2-main

	.type	A,@object               # @A
	.comm	A,9437184,16
	.type	B,@object               # @B
	.comm	B,9437184,16
	.type	.L.str,@object          # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	 "%lf "
	.size	.L.str, 5

	.type	C,@object               # @C
	.comm	C,9437184,16

	.section	".note.GNU-stack","",@progbits
