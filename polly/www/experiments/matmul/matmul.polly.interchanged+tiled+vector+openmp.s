	.file	"matmul.polly.interchanged+tiled+vector+openmp.ll"
	.text
	.globl	init_array
	.align	16, 0x90
	.type	init_array,@function
init_array:                             # @init_array
# BB#0:                                 # %pollyBB
	pushq	%rbx
	subq	$16, %rsp
	movq	$A, (%rsp)
	movq	$B, 8(%rsp)
	movl	$init_array.omp_subfn, %edi
	leaq	(%rsp), %rbx
	xorl	%edx, %edx
	xorl	%ecx, %ecx
	movl	$1536, %r8d             # imm = 0x600
	movl	$1, %r9d
	movq	%rbx, %rsi
	callq	GOMP_parallel_loop_runtime_start
	movq	%rbx, %rdi
	callq	init_array.omp_subfn
	callq	GOMP_parallel_end
	addq	$16, %rsp
	popq	%rbx
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

	.globl	main
	.align	16, 0x90
	.type	main,@function
main:                                   # @main
# BB#0:                                 # %pollyBB
	pushq	%rbp
	movq	%rsp, %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$56, %rsp
	movq	$A, -72(%rbp)
	movq	$B, -64(%rbp)
	movl	$init_array.omp_subfn, %edi
	leaq	-72(%rbp), %rbx
	movq	%rbx, %rsi
	xorl	%edx, %edx
	xorl	%ecx, %ecx
	movl	$1536, %r8d             # imm = 0x600
	movl	$1, %r9d
	callq	GOMP_parallel_loop_runtime_start
	movq	%rbx, %rdi
	callq	init_array.omp_subfn
	callq	GOMP_parallel_end
	movl	$main.omp_subfn, %edi
	leaq	-96(%rbp), %rsi
	movq	$C, -96(%rbp)
	movq	$A, -88(%rbp)
	movq	$B, -80(%rbp)
	xorl	%edx, %edx
	xorl	%ecx, %ecx
	movl	$1536, %r8d             # imm = 0x600
	movl	$1, %r9d
	callq	GOMP_parallel_loop_runtime_start
	leaq	-48(%rbp), %rdi
	leaq	-56(%rbp), %rsi
	callq	GOMP_loop_runtime_next
	testb	$1, %al
	je	.LBB2_6
# BB#1:
	leaq	-48(%rbp), %rbx
	leaq	-56(%rbp), %r14
	.align	16, 0x90
.LBB2_3:                                # %omp.loadIVBounds.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB2_5 Depth 2
	movq	-56(%rbp), %r15
	decq	%r15
	movq	-48(%rbp), %r12
	cmpq	%r15, %r12
	jg	.LBB2_2
# BB#4:                                 # %polly.loop_header2.preheader.lr.ph.i
                                        #   in Loop: Header=BB2_3 Depth=1
	leaq	(%r12,%r12,2), %rax
	shlq	$11, %rax
	leaq	C(%rax), %r13
	.align	16, 0x90
.LBB2_5:                                # %polly.loop_header2.preheader.i
                                        #   Parent Loop BB2_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movq	%r13, %rdi
	xorl	%esi, %esi
	movl	$6144, %edx             # imm = 0x1800
	callq	memset
	addq	$6144, %r13             # imm = 0x1800
	incq	%r12
	cmpq	%r15, %r12
	jle	.LBB2_5
.LBB2_2:                                # %omp.checkNext.loopexit.i
                                        #   in Loop: Header=BB2_3 Depth=1
	movq	%rbx, %rdi
	movq	%r14, %rsi
	callq	GOMP_loop_runtime_next
	testb	$1, %al
	jne	.LBB2_3
.LBB2_6:                                # %main.omp_subfn.exit
	callq	GOMP_loop_end_nowait
	callq	GOMP_parallel_end
	movq	%rsp, %rax
	leaq	-32(%rax), %rbx
	movl	$main.omp_subfn1, %edi
	xorl	%ecx, %ecx
	movl	$1536, %r8d             # imm = 0x600
	movl	$64, %r9d
	movq	%rbx, %rsp
	movq	$C, -32(%rax)
	movq	$A, -24(%rax)
	movq	$B, -16(%rax)
	movq	%rbx, %rsi
	xorl	%edx, %edx
	callq	GOMP_parallel_loop_runtime_start
	movq	%rbx, %rdi
	callq	main.omp_subfn1
	callq	GOMP_parallel_end
	xorl	%eax, %eax
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	ret
.Ltmp2:
	.size	main, .Ltmp2-main

	.section	.rodata.cst8,"aM",@progbits,8
	.align	8
.LCPI3_0:
	.quad	4602678819172646912     # double 5.000000e-01
	.text
	.align	16, 0x90
	.type	init_array.omp_subfn,@function
init_array.omp_subfn:                   # @init_array.omp_subfn
.Leh_func_begin3:
.Ltmp6:
	.cfi_startproc
# BB#0:                                 # %omp.setup
	pushq	%r14
.Ltmp7:
	.cfi_def_cfa_offset 16
	pushq	%rbx
.Ltmp8:
	.cfi_def_cfa_offset 24
	subq	$24, %rsp
.Ltmp9:
	.cfi_def_cfa_offset 48
.Ltmp10:
	.cfi_offset 3, -24
.Ltmp11:
	.cfi_offset 14, -16
	leaq	16(%rsp), %rdi
	leaq	8(%rsp), %rsi
	callq	GOMP_loop_runtime_next
	testb	$1, %al
	je	.LBB3_2
# BB#1:
	leaq	16(%rsp), %rbx
	leaq	8(%rsp), %r14
	jmp	.LBB3_4
.LBB3_2:                                # %omp.exit
	callq	GOMP_loop_end_nowait
	addq	$24, %rsp
	popq	%rbx
	popq	%r14
	ret
	.align	16, 0x90
.LBB3_3:                                # %omp.checkNext.loopexit
                                        #   in Loop: Header=BB3_4 Depth=1
	movq	%rbx, %rdi
	movq	%r14, %rsi
	callq	GOMP_loop_runtime_next
	testb	$1, %al
	je	.LBB3_2
.LBB3_4:                                # %omp.loadIVBounds
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_7 Depth 2
                                        #       Child Loop BB3_8 Depth 3
	movq	8(%rsp), %rax
	decq	%rax
	movq	16(%rsp), %rcx
	cmpq	%rax, %rcx
	jg	.LBB3_3
# BB#5:                                 # %polly.loop_header2.preheader.lr.ph
                                        #   in Loop: Header=BB3_4 Depth=1
	movq	%rcx, %rdx
	shlq	$11, %rdx
	leaq	(%rdx,%rdx,2), %rdx
	jmp	.LBB3_7
	.align	16, 0x90
.LBB3_6:                                # %polly.loop_header.loopexit
                                        #   in Loop: Header=BB3_7 Depth=2
	addq	$6144, %rdx             # imm = 0x1800
	incq	%rcx
	cmpq	%rax, %rcx
	jg	.LBB3_3
.LBB3_7:                                # %polly.loop_header2.preheader
                                        #   Parent Loop BB3_4 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB3_8 Depth 3
	movq	$-1536, %rsi            # imm = 0xFFFFFFFFFFFFFA00
	xorl	%edi, %edi
	.align	16, 0x90
.LBB3_8:                                # %polly.loop_body3
                                        #   Parent Loop BB3_4 Depth=1
                                        #     Parent Loop BB3_7 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movl	%edi, %r8d
	sarl	$31, %r8d
	shrl	$22, %r8d
	addl	%edi, %r8d
	andl	$-1024, %r8d            # imm = 0xFFFFFFFFFFFFFC00
	negl	%r8d
	leal	1(%rdi,%r8), %r8d
	cvtsi2sd	%r8d, %xmm0
	mulsd	.LCPI3_0(%rip), %xmm0
	cvtsd2ss	%xmm0, %xmm0
	movss	%xmm0, A+6144(%rdx,%rsi,4)
	movss	%xmm0, B+6144(%rdx,%rsi,4)
	addl	%ecx, %edi
	incq	%rsi
	jne	.LBB3_8
	jmp	.LBB3_6
.Ltmp12:
	.size	init_array.omp_subfn, .Ltmp12-init_array.omp_subfn
.Ltmp13:
	.cfi_endproc
.Leh_func_end3:

	.align	16, 0x90
	.type	main.omp_subfn,@function
main.omp_subfn:                         # @main.omp_subfn
.Leh_func_begin4:
.Ltmp20:
	.cfi_startproc
# BB#0:                                 # %omp.setup
	pushq	%r15
.Ltmp21:
	.cfi_def_cfa_offset 16
	pushq	%r14
.Ltmp22:
	.cfi_def_cfa_offset 24
	pushq	%r13
.Ltmp23:
	.cfi_def_cfa_offset 32
	pushq	%r12
.Ltmp24:
	.cfi_def_cfa_offset 40
	pushq	%rbx
.Ltmp25:
	.cfi_def_cfa_offset 48
	subq	$16, %rsp
.Ltmp26:
	.cfi_def_cfa_offset 64
.Ltmp27:
	.cfi_offset 3, -48
.Ltmp28:
	.cfi_offset 12, -40
.Ltmp29:
	.cfi_offset 13, -32
.Ltmp30:
	.cfi_offset 14, -24
.Ltmp31:
	.cfi_offset 15, -16
	leaq	8(%rsp), %rdi
	leaq	(%rsp), %rsi
	callq	GOMP_loop_runtime_next
	testb	$1, %al
	je	.LBB4_2
# BB#1:
	leaq	8(%rsp), %rbx
	leaq	(%rsp), %r14
	jmp	.LBB4_4
.LBB4_2:                                # %omp.exit
	callq	GOMP_loop_end_nowait
	addq	$16, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	ret
	.align	16, 0x90
.LBB4_3:                                # %omp.checkNext.loopexit
                                        #   in Loop: Header=BB4_4 Depth=1
	movq	%rbx, %rdi
	movq	%r14, %rsi
	callq	GOMP_loop_runtime_next
	testb	$1, %al
	je	.LBB4_2
.LBB4_4:                                # %omp.loadIVBounds
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB4_6 Depth 2
	movq	(%rsp), %r15
	decq	%r15
	movq	8(%rsp), %r12
	cmpq	%r15, %r12
	jg	.LBB4_3
# BB#5:                                 # %polly.loop_header2.preheader.lr.ph
                                        #   in Loop: Header=BB4_4 Depth=1
	leaq	(%r12,%r12,2), %rax
	shlq	$11, %rax
	leaq	C(%rax), %r13
	.align	16, 0x90
.LBB4_6:                                # %polly.loop_header2.preheader
                                        #   Parent Loop BB4_4 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movq	%r13, %rdi
	xorl	%esi, %esi
	movl	$6144, %edx             # imm = 0x1800
	callq	memset
	addq	$6144, %r13             # imm = 0x1800
	incq	%r12
	cmpq	%r15, %r12
	jle	.LBB4_6
	jmp	.LBB4_3
.Ltmp32:
	.size	main.omp_subfn, .Ltmp32-main.omp_subfn
.Ltmp33:
	.cfi_endproc
.Leh_func_end4:

	.align	16, 0x90
	.type	main.omp_subfn1,@function
main.omp_subfn1:                        # @main.omp_subfn1
.Leh_func_begin5:
.Ltmp41:
	.cfi_startproc
# BB#0:                                 # %omp.setup
	pushq	%rbp
.Ltmp42:
	.cfi_def_cfa_offset 16
	pushq	%r15
.Ltmp43:
	.cfi_def_cfa_offset 24
	pushq	%r14
.Ltmp44:
	.cfi_def_cfa_offset 32
	pushq	%r13
.Ltmp45:
	.cfi_def_cfa_offset 40
	pushq	%r12
.Ltmp46:
	.cfi_def_cfa_offset 48
	pushq	%rbx
.Ltmp47:
	.cfi_def_cfa_offset 56
	subq	$40, %rsp
.Ltmp48:
	.cfi_def_cfa_offset 96
.Ltmp49:
	.cfi_offset 3, -56
.Ltmp50:
	.cfi_offset 12, -48
.Ltmp51:
	.cfi_offset 13, -40
.Ltmp52:
	.cfi_offset 14, -32
.Ltmp53:
	.cfi_offset 15, -24
.Ltmp54:
	.cfi_offset 6, -16
	leaq	32(%rsp), %rdi
	leaq	24(%rsp), %rsi
	jmp	.LBB5_1
	.align	16, 0x90
.LBB5_4:                                # %omp.loadIVBounds
                                        #   in Loop: Header=BB5_1 Depth=1
	movq	24(%rsp), %rax
	decq	%rax
	movq	%rax, (%rsp)            # 8-byte Spill
	movq	32(%rsp), %rcx
	cmpq	%rax, %rcx
	jg	.LBB5_3
# BB#5:                                 # %polly.loop_header2.preheader.lr.ph
                                        #   in Loop: Header=BB5_1 Depth=1
	leaq	(%rcx,%rcx,2), %rax
	movq	%rcx, %rdx
	shlq	$9, %rdx
	leaq	(%rdx,%rdx,2), %rdx
	movq	%rdx, 16(%rsp)          # 8-byte Spill
	shlq	$11, %rax
	leaq	A(%rax), %rax
	movq	%rax, 8(%rsp)           # 8-byte Spill
	jmp	.LBB5_7
	.align	16, 0x90
.LBB5_6:                                # %polly.loop_header.loopexit
                                        #   in Loop: Header=BB5_7 Depth=2
	addq	$98304, 16(%rsp)        # 8-byte Folded Spill
                                        # imm = 0x18000
	addq	$393216, 8(%rsp)        # 8-byte Folded Spill
                                        # imm = 0x60000
	addq	$64, %rcx
	cmpq	(%rsp), %rcx            # 8-byte Folded Reload
	jg	.LBB5_3
.LBB5_7:                                # %polly.loop_header2.preheader
                                        #   Parent Loop BB5_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB5_9 Depth 3
                                        #         Child Loop BB5_11 Depth 4
                                        #           Child Loop BB5_14 Depth 5
                                        #             Child Loop BB5_18 Depth 6
                                        #               Child Loop BB5_19 Depth 7
	leaq	63(%rcx), %rax
	xorl	%edx, %edx
	jmp	.LBB5_9
	.align	16, 0x90
.LBB5_8:                                # %polly.loop_header2.loopexit
                                        #   in Loop: Header=BB5_9 Depth=3
	addq	$64, %rdx
	cmpq	$1536, %rdx             # imm = 0x600
	je	.LBB5_6
.LBB5_9:                                # %polly.loop_header7.preheader
                                        #   Parent Loop BB5_1 Depth=1
                                        #     Parent Loop BB5_7 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB5_11 Depth 4
                                        #           Child Loop BB5_14 Depth 5
                                        #             Child Loop BB5_18 Depth 6
                                        #               Child Loop BB5_19 Depth 7
	movq	16(%rsp), %rsi          # 8-byte Reload
	leaq	(%rsi,%rdx), %rsi
	leaq	63(%rdx), %rdi
	xorl	%r8d, %r8d
	movq	8(%rsp), %r9            # 8-byte Reload
	movq	%rdx, %r10
	jmp	.LBB5_11
	.align	16, 0x90
.LBB5_10:                               # %polly.loop_header7.loopexit
                                        #   in Loop: Header=BB5_11 Depth=4
	addq	$256, %r9               # imm = 0x100
	addq	$98304, %r10            # imm = 0x18000
	addq	$64, %r8
	cmpq	$1536, %r8              # imm = 0x600
	je	.LBB5_8
.LBB5_11:                               # %polly.loop_body8
                                        #   Parent Loop BB5_1 Depth=1
                                        #     Parent Loop BB5_7 Depth=2
                                        #       Parent Loop BB5_9 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB5_14 Depth 5
                                        #             Child Loop BB5_18 Depth 6
                                        #               Child Loop BB5_19 Depth 7
	movabsq	$9223372036854775744, %r11 # imm = 0x7FFFFFFFFFFFFFC0
	cmpq	%r11, %rcx
	jg	.LBB5_10
# BB#12:                                # %polly.loop_body13.lr.ph
                                        #   in Loop: Header=BB5_11 Depth=4
	leaq	63(%r8), %r11
	movq	%rcx, %rbx
	movq	%rsi, %r14
	movq	%r9, %r15
	jmp	.LBB5_14
	.align	16, 0x90
.LBB5_13:                               # %polly.loop_header12.loopexit
                                        #   in Loop: Header=BB5_14 Depth=5
	addq	$1536, %r14             # imm = 0x600
	addq	$6144, %r15             # imm = 0x1800
	incq	%rbx
	cmpq	%rax, %rbx
	jg	.LBB5_10
.LBB5_14:                               # %polly.loop_body13
                                        #   Parent Loop BB5_1 Depth=1
                                        #     Parent Loop BB5_7 Depth=2
                                        #       Parent Loop BB5_9 Depth=3
                                        #         Parent Loop BB5_11 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB5_18 Depth 6
                                        #               Child Loop BB5_19 Depth 7
	cmpq	%r11, %r8
	jg	.LBB5_13
# BB#15:                                # %polly.loop_body13
                                        #   in Loop: Header=BB5_14 Depth=5
	cmpq	%rdi, %rdx
	jg	.LBB5_13
# BB#16:                                # %polly.loop_body23.lr.ph.preheader
                                        #   in Loop: Header=BB5_14 Depth=5
	xorl	%r12d, %r12d
	movq	%r10, %r13
	jmp	.LBB5_18
	.align	16, 0x90
.LBB5_17:                               # %polly.loop_header17.loopexit
                                        #   in Loop: Header=BB5_18 Depth=6
	addq	$1536, %r13             # imm = 0x600
	incq	%r12
	cmpq	$64, %r12
	je	.LBB5_13
.LBB5_18:                               # %polly.loop_body23.lr.ph
                                        #   Parent Loop BB5_1 Depth=1
                                        #     Parent Loop BB5_7 Depth=2
                                        #       Parent Loop BB5_9 Depth=3
                                        #         Parent Loop BB5_11 Depth=4
                                        #           Parent Loop BB5_14 Depth=5
                                        # =>          This Loop Header: Depth=6
                                        #               Child Loop BB5_19 Depth 7
	movss	(%r15,%r12,4), %xmm0
	pshufd	$0, %xmm0, %xmm0        # xmm0 = xmm0[0,0,0,0]
	xorl	%ebp, %ebp
	.align	16, 0x90
.LBB5_19:                               # %polly.loop_body23
                                        #   Parent Loop BB5_1 Depth=1
                                        #     Parent Loop BB5_7 Depth=2
                                        #       Parent Loop BB5_9 Depth=3
                                        #         Parent Loop BB5_11 Depth=4
                                        #           Parent Loop BB5_14 Depth=5
                                        #             Parent Loop BB5_18 Depth=6
                                        # =>            This Inner Loop Header: Depth=7
	movaps	B(%rbp,%r13,4), %xmm1
	mulps	%xmm0, %xmm1
	addps	C(%rbp,%r14,4), %xmm1
	movaps	%xmm1, C(%rbp,%r14,4)
	addq	$16, %rbp
	cmpq	$256, %rbp              # imm = 0x100
	jne	.LBB5_19
	jmp	.LBB5_17
.LBB5_3:                                # %omp.checkNext.loopexit
                                        #   in Loop: Header=BB5_1 Depth=1
	leaq	32(%rsp), %rax
	movq	%rax, %rdi
	leaq	24(%rsp), %rax
	movq	%rax, %rsi
.LBB5_1:                                # %omp.setup
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB5_7 Depth 2
                                        #       Child Loop BB5_9 Depth 3
                                        #         Child Loop BB5_11 Depth 4
                                        #           Child Loop BB5_14 Depth 5
                                        #             Child Loop BB5_18 Depth 6
                                        #               Child Loop BB5_19 Depth 7
	callq	GOMP_loop_runtime_next
	testb	$1, %al
	jne	.LBB5_4
# BB#2:                                 # %omp.exit
	callq	GOMP_loop_end_nowait
	addq	$40, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	ret
.Ltmp55:
	.size	main.omp_subfn1, .Ltmp55-main.omp_subfn1
.Ltmp56:
	.cfi_endproc
.Leh_func_end5:

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
