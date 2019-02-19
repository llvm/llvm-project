	.file	"matmul.polly.interchanged+tiled+vector+openmp.ll"
	.section	.rodata.cst8,"aM",@progbits,8
	.align	8
.LCPI0_0:
	.quad	4602678819172646912     # double 0.5
	.text
	.globl	init_array
	.align	16, 0x90
	.type	init_array,@function
init_array:                             # @init_array
	.cfi_startproc
# BB#0:                                 # %entry
	pushq	%rbp
.Ltmp3:
	.cfi_def_cfa_offset 16
.Ltmp4:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
.Ltmp5:
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%rbx
	subq	$24, %rsp
.Ltmp6:
	.cfi_offset %rbx, -40
.Ltmp7:
	.cfi_offset %r14, -32
.Ltmp8:
	.cfi_offset %r15, -24
	leaq	-32(%rbp), %rsi
	movl	$init_array.omp_subfn, %edi
	xorl	%edx, %edx
	xorl	%ecx, %ecx
	movl	$1536, %r8d             # imm = 0x600
	movl	$1, %r9d
	callq	GOMP_parallel_loop_runtime_start
	leaq	-40(%rbp), %rdi
	leaq	-48(%rbp), %rsi
	callq	GOMP_loop_runtime_next
	testb	%al, %al
	je	.LBB0_4
# BB#1:
	leaq	-40(%rbp), %r14
	leaq	-48(%rbp), %r15
	vmovsd	.LCPI0_0(%rip), %xmm1
	.align	16, 0x90
.LBB0_2:                                # %omp.loadIVBounds.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_8 Depth 2
                                        #       Child Loop BB0_5 Depth 3
	movq	-48(%rbp), %r8
	leaq	-1(%r8), %rcx
	movq	-40(%rbp), %rax
	cmpq	%rcx, %rax
	jg	.LBB0_3
# BB#7:                                 # %polly.loop_preheader4.preheader.i
                                        #   in Loop: Header=BB0_2 Depth=1
	addq	$-2, %r8
	.align	16, 0x90
.LBB0_8:                                # %polly.loop_preheader4.i
                                        #   Parent Loop BB0_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_5 Depth 3
	xorl	%edx, %edx
	.align	16, 0x90
.LBB0_5:                                # %polly.loop_header3.i
                                        #   Parent Loop BB0_2 Depth=1
                                        #     Parent Loop BB0_8 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movl	%edx, %esi
	imull	%eax, %esi
	movl	%esi, %edi
	sarl	$31, %edi
	shrl	$22, %edi
	addl	%esi, %edi
	andl	$-1024, %edi            # imm = 0xFFFFFFFFFFFFFC00
	negl	%edi
	movq	%rax, %rcx
	shlq	$11, %rcx
	leal	1(%rsi,%rdi), %ebx
	leaq	(%rcx,%rcx,2), %rdi
	leaq	1(%rdx), %rsi
	cmpq	$1536, %rsi             # imm = 0x600
	vcvtsi2sdl	%ebx, %xmm0, %xmm0
	vmulsd	%xmm1, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vmovss	%xmm0, A(%rdi,%rdx,4)
	vmovss	%xmm0, B(%rdi,%rdx,4)
	movq	%rsi, %rdx
	jne	.LBB0_5
# BB#6:                                 # %polly.loop_exit5.i
                                        #   in Loop: Header=BB0_8 Depth=2
	cmpq	%r8, %rax
	leaq	1(%rax), %rax
	jle	.LBB0_8
.LBB0_3:                                # %omp.checkNext.backedge.i
                                        #   in Loop: Header=BB0_2 Depth=1
	movq	%r14, %rdi
	movq	%r15, %rsi
	callq	GOMP_loop_runtime_next
	vmovsd	.LCPI0_0(%rip), %xmm1
	testb	%al, %al
	jne	.LBB0_2
.LBB0_4:                                # %init_array.omp_subfn.exit
	callq	GOMP_loop_end_nowait
	callq	GOMP_parallel_end
	addq	$24, %rsp
	popq	%rbx
	popq	%r14
	popq	%r15
	popq	%rbp
	ret
.Ltmp9:
	.size	init_array, .Ltmp9-init_array
	.cfi_endproc

	.globl	print_array
	.align	16, 0x90
	.type	print_array,@function
print_array:                            # @print_array
	.cfi_startproc
# BB#0:                                 # %entry
	pushq	%rbp
.Ltmp13:
	.cfi_def_cfa_offset 16
.Ltmp14:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
.Ltmp15:
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r12
	pushq	%rbx
.Ltmp16:
	.cfi_offset %rbx, -48
.Ltmp17:
	.cfi_offset %r12, -40
.Ltmp18:
	.cfi_offset %r14, -32
.Ltmp19:
	.cfi_offset %r15, -24
	xorl	%r14d, %r14d
	movl	$C, %r15d
	.align	16, 0x90
.LBB1_1:                                # %for.cond1.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB1_2 Depth 2
	movq	stdout(%rip), %rax
	movq	%r15, %r12
	xorl	%ebx, %ebx
	.align	16, 0x90
.LBB1_2:                                # %for.body3
                                        #   Parent Loop BB1_1 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vmovss	(%r12), %xmm0
	vcvtss2sd	%xmm0, %xmm0, %xmm0
	movq	%rax, %rdi
	movl	$.L.str, %esi
	movb	$1, %al
	callq	fprintf
	movslq	%ebx, %rax
	imulq	$1717986919, %rax, %rcx # imm = 0x66666667
	movq	%rcx, %rdx
	shrq	$63, %rdx
	sarq	$37, %rcx
	addl	%edx, %ecx
	imull	$80, %ecx, %ecx
	subl	%ecx, %eax
	cmpl	$79, %eax
	jne	.LBB1_4
# BB#3:                                 # %if.then
                                        #   in Loop: Header=BB1_2 Depth=2
	movq	stdout(%rip), %rsi
	movl	$10, %edi
	callq	fputc
.LBB1_4:                                # %for.inc
                                        #   in Loop: Header=BB1_2 Depth=2
	addq	$4, %r12
	incq	%rbx
	movq	stdout(%rip), %rax
	cmpq	$1536, %rbx             # imm = 0x600
	jne	.LBB1_2
# BB#5:                                 # %for.end
                                        #   in Loop: Header=BB1_1 Depth=1
	movl	$10, %edi
	movq	%rax, %rsi
	callq	fputc
	addq	$6144, %r15             # imm = 0x1800
	incq	%r14
	cmpq	$1536, %r14             # imm = 0x600
	jne	.LBB1_1
# BB#6:                                 # %for.end12
	popq	%rbx
	popq	%r12
	popq	%r14
	popq	%r15
	popq	%rbp
	ret
.Ltmp20:
	.size	print_array, .Ltmp20-print_array
	.cfi_endproc

	.globl	main
	.align	16, 0x90
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# BB#0:                                 # %entry
	pushq	%rbp
.Ltmp24:
	.cfi_def_cfa_offset 16
.Ltmp25:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
.Ltmp26:
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$24, %rsp
.Ltmp27:
	.cfi_offset %rbx, -56
.Ltmp28:
	.cfi_offset %r12, -48
.Ltmp29:
	.cfi_offset %r13, -40
.Ltmp30:
	.cfi_offset %r14, -32
.Ltmp31:
	.cfi_offset %r15, -24
	callq	init_array
	leaq	-48(%rbp), %rsi
	movl	$main.omp_subfn, %edi
	xorl	%edx, %edx
	xorl	%ecx, %ecx
	movl	$1536, %r8d             # imm = 0x600
	movl	$1, %r9d
	callq	GOMP_parallel_loop_runtime_start
	leaq	-56(%rbp), %rdi
	leaq	-64(%rbp), %rsi
	callq	GOMP_loop_runtime_next
	testb	%al, %al
	je	.LBB2_4
# BB#1:
	leaq	-56(%rbp), %r14
	leaq	-64(%rbp), %r15
	.align	16, 0x90
.LBB2_2:                                # %omp.loadIVBounds.i
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB2_6 Depth 2
	movq	-64(%rbp), %r12
	leaq	-1(%r12), %rcx
	movq	-56(%rbp), %rax
	cmpq	%rcx, %rax
	jg	.LBB2_3
# BB#5:                                 # %polly.loop_preheader4.preheader.i
                                        #   in Loop: Header=BB2_2 Depth=1
	addq	$-2, %r12
	leaq	(%rax,%rax,2), %rcx
	leaq	-1(%rax), %r13
	shlq	$11, %rcx
	leaq	C(%rcx), %rbx
	.align	16, 0x90
.LBB2_6:                                # %polly.loop_preheader4.i
                                        #   Parent Loop BB2_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movq	%rbx, %rdi
	xorl	%esi, %esi
	movl	$6144, %edx             # imm = 0x1800
	callq	memset
	addq	$6144, %rbx             # imm = 0x1800
	incq	%r13
	cmpq	%r12, %r13
	jle	.LBB2_6
.LBB2_3:                                # %omp.checkNext.backedge.i
                                        #   in Loop: Header=BB2_2 Depth=1
	movq	%r14, %rdi
	movq	%r15, %rsi
	callq	GOMP_loop_runtime_next
	testb	%al, %al
	jne	.LBB2_2
.LBB2_4:                                # %main.omp_subfn.exit
	callq	GOMP_loop_end_nowait
	callq	GOMP_parallel_end
	leaq	-48(%rbp), %rbx
	movl	$main.omp_subfn1, %edi
	movq	%rbx, %rsi
	xorl	%edx, %edx
	xorl	%ecx, %ecx
	movl	$1536, %r8d             # imm = 0x600
	movl	$64, %r9d
	callq	GOMP_parallel_loop_runtime_start
	movq	%rbx, %rdi
	callq	main.omp_subfn1
	callq	GOMP_parallel_end
	xorl	%eax, %eax
	addq	$24, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	ret
.Ltmp32:
	.size	main, .Ltmp32-main
	.cfi_endproc

	.section	.rodata.cst8,"aM",@progbits,8
	.align	8
.LCPI3_0:
	.quad	4602678819172646912     # double 0.5
	.text
	.align	16, 0x90
	.type	init_array.omp_subfn,@function
init_array.omp_subfn:                   # @init_array.omp_subfn
	.cfi_startproc
# BB#0:                                 # %omp.setup
	pushq	%rbp
.Ltmp36:
	.cfi_def_cfa_offset 16
.Ltmp37:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
.Ltmp38:
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%rbx
	subq	$24, %rsp
.Ltmp39:
	.cfi_offset %rbx, -40
.Ltmp40:
	.cfi_offset %r14, -32
.Ltmp41:
	.cfi_offset %r15, -24
	leaq	-32(%rbp), %rdi
	leaq	-40(%rbp), %rsi
	callq	GOMP_loop_runtime_next
	testb	%al, %al
	je	.LBB3_4
# BB#1:
	leaq	-32(%rbp), %r14
	leaq	-40(%rbp), %r15
	vmovsd	.LCPI3_0(%rip), %xmm1
	.align	16, 0x90
.LBB3_2:                                # %omp.loadIVBounds
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB3_8 Depth 2
                                        #       Child Loop BB3_5 Depth 3
	movq	-40(%rbp), %r8
	leaq	-1(%r8), %rcx
	movq	-32(%rbp), %rax
	cmpq	%rcx, %rax
	jg	.LBB3_3
# BB#7:                                 # %polly.loop_preheader4.preheader
                                        #   in Loop: Header=BB3_2 Depth=1
	addq	$-2, %r8
	.align	16, 0x90
.LBB3_8:                                # %polly.loop_preheader4
                                        #   Parent Loop BB3_2 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB3_5 Depth 3
	xorl	%edx, %edx
	.align	16, 0x90
.LBB3_5:                                # %polly.loop_header3
                                        #   Parent Loop BB3_2 Depth=1
                                        #     Parent Loop BB3_8 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movl	%edx, %esi
	imull	%eax, %esi
	movl	%esi, %edi
	sarl	$31, %edi
	shrl	$22, %edi
	addl	%esi, %edi
	andl	$-1024, %edi            # imm = 0xFFFFFFFFFFFFFC00
	negl	%edi
	movq	%rax, %rcx
	shlq	$11, %rcx
	leal	1(%rsi,%rdi), %ebx
	leaq	(%rcx,%rcx,2), %rdi
	leaq	1(%rdx), %rsi
	cmpq	$1536, %rsi             # imm = 0x600
	vcvtsi2sdl	%ebx, %xmm0, %xmm0
	vmulsd	%xmm1, %xmm0, %xmm0
	vcvtsd2ss	%xmm0, %xmm0, %xmm0
	vmovss	%xmm0, A(%rdi,%rdx,4)
	vmovss	%xmm0, B(%rdi,%rdx,4)
	movq	%rsi, %rdx
	jne	.LBB3_5
# BB#6:                                 # %polly.loop_exit5
                                        #   in Loop: Header=BB3_8 Depth=2
	cmpq	%r8, %rax
	leaq	1(%rax), %rax
	jle	.LBB3_8
.LBB3_3:                                # %omp.checkNext.backedge
                                        #   in Loop: Header=BB3_2 Depth=1
	movq	%r14, %rdi
	movq	%r15, %rsi
	callq	GOMP_loop_runtime_next
	vmovsd	.LCPI3_0(%rip), %xmm1
	testb	%al, %al
	jne	.LBB3_2
.LBB3_4:                                # %omp.exit
	callq	GOMP_loop_end_nowait
	addq	$24, %rsp
	popq	%rbx
	popq	%r14
	popq	%r15
	popq	%rbp
	ret
.Ltmp42:
	.size	init_array.omp_subfn, .Ltmp42-init_array.omp_subfn
	.cfi_endproc

	.align	16, 0x90
	.type	main.omp_subfn,@function
main.omp_subfn:                         # @main.omp_subfn
	.cfi_startproc
# BB#0:                                 # %omp.setup
	pushq	%rbp
.Ltmp46:
	.cfi_def_cfa_offset 16
.Ltmp47:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
.Ltmp48:
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$24, %rsp
.Ltmp49:
	.cfi_offset %rbx, -56
.Ltmp50:
	.cfi_offset %r12, -48
.Ltmp51:
	.cfi_offset %r13, -40
.Ltmp52:
	.cfi_offset %r14, -32
.Ltmp53:
	.cfi_offset %r15, -24
	leaq	-48(%rbp), %rdi
	leaq	-56(%rbp), %rsi
	callq	GOMP_loop_runtime_next
	testb	%al, %al
	je	.LBB4_4
# BB#1:
	leaq	-48(%rbp), %r14
	leaq	-56(%rbp), %r15
	.align	16, 0x90
.LBB4_2:                                # %omp.loadIVBounds
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB4_6 Depth 2
	movq	-56(%rbp), %r12
	leaq	-1(%r12), %rcx
	movq	-48(%rbp), %rax
	cmpq	%rcx, %rax
	jg	.LBB4_3
# BB#5:                                 # %polly.loop_preheader4.preheader
                                        #   in Loop: Header=BB4_2 Depth=1
	addq	$-2, %r12
	leaq	(%rax,%rax,2), %rcx
	leaq	-1(%rax), %r13
	shlq	$11, %rcx
	leaq	C(%rcx), %rbx
	.align	16, 0x90
.LBB4_6:                                # %polly.loop_preheader4
                                        #   Parent Loop BB4_2 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movq	%rbx, %rdi
	xorl	%esi, %esi
	movl	$6144, %edx             # imm = 0x1800
	callq	memset
	addq	$6144, %rbx             # imm = 0x1800
	incq	%r13
	cmpq	%r12, %r13
	jle	.LBB4_6
.LBB4_3:                                # %omp.checkNext.backedge
                                        #   in Loop: Header=BB4_2 Depth=1
	movq	%r14, %rdi
	movq	%r15, %rsi
	callq	GOMP_loop_runtime_next
	testb	%al, %al
	jne	.LBB4_2
.LBB4_4:                                # %omp.exit
	callq	GOMP_loop_end_nowait
	addq	$24, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	ret
.Ltmp54:
	.size	main.omp_subfn, .Ltmp54-main.omp_subfn
	.cfi_endproc

	.align	16, 0x90
	.type	main.omp_subfn1,@function
main.omp_subfn1:                        # @main.omp_subfn1
	.cfi_startproc
# BB#0:                                 # %omp.setup
	pushq	%rbp
.Ltmp58:
	.cfi_def_cfa_offset 16
.Ltmp59:
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
.Ltmp60:
	.cfi_def_cfa_register %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$72, %rsp
.Ltmp61:
	.cfi_offset %rbx, -56
.Ltmp62:
	.cfi_offset %r12, -48
.Ltmp63:
	.cfi_offset %r13, -40
.Ltmp64:
	.cfi_offset %r14, -32
.Ltmp65:
	.cfi_offset %r15, -24
	jmp	.LBB5_1
	.align	16, 0x90
.LBB5_2:                                # %omp.loadIVBounds
                                        #   in Loop: Header=BB5_1 Depth=1
	movq	-56(%rbp), %rax
	movq	%rax, -112(%rbp)        # 8-byte Spill
	leaq	-1(%rax), %rax
	movq	-48(%rbp), %rcx
	cmpq	%rax, %rcx
	jg	.LBB5_1
# BB#3:                                 # %polly.loop_preheader4.preheader
                                        #   in Loop: Header=BB5_1 Depth=1
	leaq	-1(%rcx), %rax
	movq	%rax, -88(%rbp)         # 8-byte Spill
	addq	$-65, -112(%rbp)        # 8-byte Folded Spill
	movq	%rcx, %rax
	shlq	$9, %rax
	leaq	(%rax,%rax,2), %rax
	leaq	C+16(,%rax,4), %rax
	movq	%rax, -104(%rbp)        # 8-byte Spill
	.align	16, 0x90
.LBB5_7:                                # %polly.loop_preheader4
                                        #   Parent Loop BB5_1 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB5_8 Depth 3
                                        #         Child Loop BB5_9 Depth 4
                                        #           Child Loop BB5_12 Depth 5
                                        #             Child Loop BB5_17 Depth 6
                                        #               Child Loop BB5_18 Depth 7
                                        #           Child Loop BB5_14 Depth 5
	movq	%rcx, -72(%rbp)         # 8-byte Spill
	leaq	62(%rcx), %rdi
	xorl	%edx, %edx
	.align	16, 0x90
.LBB5_8:                                # %polly.loop_preheader11
                                        #   Parent Loop BB5_1 Depth=1
                                        #     Parent Loop BB5_7 Depth=2
                                        # =>    This Loop Header: Depth=3
                                        #         Child Loop BB5_9 Depth 4
                                        #           Child Loop BB5_12 Depth 5
                                        #             Child Loop BB5_17 Depth 6
                                        #               Child Loop BB5_18 Depth 7
                                        #           Child Loop BB5_14 Depth 5
	movq	%rdx, -96(%rbp)         # 8-byte Spill
	leaq	-4(%rdx), %rcx
	movq	%rdx, %rax
	decq	%rax
	cmovsq	%rcx, %rax
	movq	%rax, %r14
	sarq	$63, %r14
	shrq	$62, %r14
	addq	%rax, %r14
	andq	$-4, %r14
	movq	%rdx, %rax
	orq	$63, %rax
	leaq	-4(%rax), %rdx
	movq	-104(%rbp), %rcx        # 8-byte Reload
	leaq	(%rcx,%r14,4), %rcx
	movq	%rcx, -80(%rbp)         # 8-byte Spill
	leaq	B+16(,%r14,4), %rbx
	leaq	4(%r14), %rcx
	movq	%rcx, -64(%rbp)         # 8-byte Spill
	xorl	%r11d, %r11d
	.align	16, 0x90
.LBB5_9:                                # %polly.loop_header10
                                        #   Parent Loop BB5_1 Depth=1
                                        #     Parent Loop BB5_7 Depth=2
                                        #       Parent Loop BB5_8 Depth=3
                                        # =>      This Loop Header: Depth=4
                                        #           Child Loop BB5_12 Depth 5
                                        #             Child Loop BB5_17 Depth 6
                                        #               Child Loop BB5_18 Depth 7
                                        #           Child Loop BB5_14 Depth 5
	movabsq	$9223372036854775744, %rcx # imm = 0x7FFFFFFFFFFFFFC0
	cmpq	%rcx, -72(%rbp)         # 8-byte Folded Reload
	jg	.LBB5_15
# BB#10:                                # %polly.loop_header17.preheader
                                        #   in Loop: Header=BB5_9 Depth=4
	movq	%r11, %r15
	orq	$63, %r15
	cmpq	%r15, %r11
	movq	-88(%rbp), %rcx         # 8-byte Reload
	jle	.LBB5_11
	.align	16, 0x90
.LBB5_14:                               # %polly.loop_exit28.us
                                        #   Parent Loop BB5_1 Depth=1
                                        #     Parent Loop BB5_7 Depth=2
                                        #       Parent Loop BB5_8 Depth=3
                                        #         Parent Loop BB5_9 Depth=4
                                        # =>        This Inner Loop Header: Depth=5
	incq	%rcx
	cmpq	%rdi, %rcx
	jle	.LBB5_14
	jmp	.LBB5_15
	.align	16, 0x90
.LBB5_11:                               #   in Loop: Header=BB5_9 Depth=4
	decq	%r15
	movq	-80(%rbp), %r13         # 8-byte Reload
	movq	-72(%rbp), %rcx         # 8-byte Reload
	.align	16, 0x90
.LBB5_12:                               # %polly.loop_header26.preheader
                                        #   Parent Loop BB5_1 Depth=1
                                        #     Parent Loop BB5_7 Depth=2
                                        #       Parent Loop BB5_8 Depth=3
                                        #         Parent Loop BB5_9 Depth=4
                                        # =>        This Loop Header: Depth=5
                                        #             Child Loop BB5_17 Depth 6
                                        #               Child Loop BB5_18 Depth 7
	cmpq	%rax, -64(%rbp)         # 8-byte Folded Reload
	movq	%rbx, %r12
	movq	%r11, %r8
	jg	.LBB5_13
	.align	16, 0x90
.LBB5_17:                               # %polly.loop_header35.preheader
                                        #   Parent Loop BB5_1 Depth=1
                                        #     Parent Loop BB5_7 Depth=2
                                        #       Parent Loop BB5_8 Depth=3
                                        #         Parent Loop BB5_9 Depth=4
                                        #           Parent Loop BB5_12 Depth=5
                                        # =>          This Loop Header: Depth=6
                                        #               Child Loop BB5_18 Depth 7
	leaq	(%rcx,%rcx,2), %rsi
	shlq	$11, %rsi
	vbroadcastss	A(%rsi,%r8,4), %xmm0
	movq	%r13, %r9
	movq	%r12, %r10
	movq	%r14, %rsi
.LBB5_18:                               # %polly.loop_header35
                                        #   Parent Loop BB5_1 Depth=1
                                        #     Parent Loop BB5_7 Depth=2
                                        #       Parent Loop BB5_8 Depth=3
                                        #         Parent Loop BB5_9 Depth=4
                                        #           Parent Loop BB5_12 Depth=5
                                        #             Parent Loop BB5_17 Depth=6
                                        # =>            This Inner Loop Header: Depth=7
	vmulps	(%r10), %xmm0, %xmm1
	vaddps	(%r9), %xmm1, %xmm1
	vmovaps	%xmm1, (%r9)
	addq	$16, %r9
	addq	$16, %r10
	addq	$4, %rsi
	cmpq	%rdx, %rsi
	jle	.LBB5_18
# BB#16:                                # %polly.loop_exit37
                                        #   in Loop: Header=BB5_17 Depth=6
	addq	$6144, %r12             # imm = 0x1800
	cmpq	%r15, %r8
	leaq	1(%r8), %r8
	jle	.LBB5_17
	.align	16, 0x90
.LBB5_13:                               # %polly.loop_exit28
                                        #   in Loop: Header=BB5_12 Depth=5
	addq	$6144, %r13             # imm = 0x1800
	cmpq	%rdi, %rcx
	leaq	1(%rcx), %rcx
	jle	.LBB5_12
	.align	16, 0x90
.LBB5_15:                               # %polly.loop_exit19
                                        #   in Loop: Header=BB5_9 Depth=4
	addq	$393216, %rbx           # imm = 0x60000
	cmpq	$1472, %r11             # imm = 0x5C0
	leaq	64(%r11), %r11
	jl	.LBB5_9
# BB#5:                                 # %polly.loop_exit12
                                        #   in Loop: Header=BB5_8 Depth=3
	movq	-96(%rbp), %rdx         # 8-byte Reload
	cmpq	$1472, %rdx             # imm = 0x5C0
	leaq	64(%rdx), %rdx
	jl	.LBB5_8
# BB#6:                                 # %polly.loop_exit5
                                        #   in Loop: Header=BB5_7 Depth=2
	addq	$64, -88(%rbp)          # 8-byte Folded Spill
	addq	$393216, -104(%rbp)     # 8-byte Folded Spill
                                        # imm = 0x60000
	movq	-72(%rbp), %rcx         # 8-byte Reload
	cmpq	-112(%rbp), %rcx        # 8-byte Folded Reload
	leaq	64(%rcx), %rcx
	jle	.LBB5_7
.LBB5_1:                                # %omp.setup
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB5_7 Depth 2
                                        #       Child Loop BB5_8 Depth 3
                                        #         Child Loop BB5_9 Depth 4
                                        #           Child Loop BB5_12 Depth 5
                                        #             Child Loop BB5_17 Depth 6
                                        #               Child Loop BB5_18 Depth 7
                                        #           Child Loop BB5_14 Depth 5
	leaq	-48(%rbp), %rdi
	leaq	-56(%rbp), %rsi
	callq	GOMP_loop_runtime_next
	testb	%al, %al
	jne	.LBB5_2
# BB#4:                                 # %omp.exit
	callq	GOMP_loop_end_nowait
	addq	$72, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	ret
.Ltmp66:
	.size	main.omp_subfn1, .Ltmp66-main.omp_subfn1
	.cfi_endproc

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
