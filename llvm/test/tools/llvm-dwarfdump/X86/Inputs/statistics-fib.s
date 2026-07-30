	.text
	.file	"fib.c"
	.globl	real_fib                # -- Begin function real_fib
	.p2align	4, 0x90
	.type	real_fib,@function
real_fib:                               # @real_fib
.Lfunc_begin0:
	.file	1 "/usr/local/google/home/cmtice/c++-tests" "fib.c"
	.loc	1 5 0                   # fib.c:5:0
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: real_fib:x <- $edi
	#DEBUG_VALUE: real_fib:answers <- $rsi
	#DEBUG_VALUE: real_fib:x <- $edi
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %rbp, -16
	.loc	1 8 7 prologue_end      # fib.c:8:7
	movslq	%edi, %rbp
	movl	(%rsi,%rbp,4), %eax
	.loc	1 8 20 is_stmt 0        # fib.c:8:20
	cmpl	$-1, %eax
.Ltmp0:
	.loc	1 8 7                   # fib.c:8:7
	je	.LBB0_1
.Ltmp1:
# %bb.2:                                # %cleanup
	#DEBUG_VALUE: real_fib:answers <- $rsi
	#DEBUG_VALUE: real_fib:x <- $edi
	.loc	1 15 1 is_stmt 1        # fib.c:15:1
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Ltmp2:
.LBB0_1:                                # %if.end
	.cfi_def_cfa_offset 32
	#DEBUG_VALUE: real_fib:answers <- $rsi
	#DEBUG_VALUE: real_fib:x <- $edi
	.loc	1 0 1 is_stmt 0         # fib.c:0:1
	movq	%rsi, %rbx
.Ltmp3:
	#DEBUG_VALUE: real_fib:answers <- $rbx
	.loc	1 11 22 is_stmt 1       # fib.c:11:22
	leal	-1(%rbp), %edi
.Ltmp4:
	.loc	1 11 12 is_stmt 0       # fib.c:11:12
	callq	real_fib
	movl	%eax, %r14d
	.loc	1 11 47                 # fib.c:11:47
	leal	-2(%rbp), %edi
	.loc	1 11 37                 # fib.c:11:37
	movq	%rbx, %rsi
	callq	real_fib
	.loc	1 11 35                 # fib.c:11:35
	addl	%r14d, %eax
.Ltmp5:
	#DEBUG_VALUE: real_fib:result <- $eax
	.loc	1 12 16 is_stmt 1       # fib.c:12:16
	movl	%eax, (%rbx,%rbp,4)
	.loc	1 15 1                  # fib.c:15:1
	popq	%rbx
.Ltmp6:
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Ltmp7:
.Lfunc_end0:
	.size	real_fib, .Lfunc_end0-real_fib
	.cfi_endproc
                                        # -- End function
	.globl	fib                     # -- Begin function fib
	.p2align	4, 0x90
	.type	fib,@function
fib:                                    # @fib
.Lfunc_begin1:
	.loc	1 19 0                  # fib.c:19:0
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: fib:x <- $edi
	movl	$-1, %eax
	#DEBUG_VALUE: fib:x <- $edi
.Ltmp8:
	.loc	1 23 9 prologue_end     # fib.c:23:9
	cmpl	$10, %edi
.Ltmp9:
	.loc	1 23 7 is_stmt 0        # fib.c:23:7
	jg	.LBB1_2
.Ltmp10:
# %bb.1:                                # %for.body.preheader
	#DEBUG_VALUE: fib:x <- $edi
	subq	$56, %rsp
	.cfi_def_cfa_offset 64
.Ltmp11:
	.loc	1 27 16 is_stmt 1       # fib.c:27:16
	pcmpeqd	%xmm0, %xmm0
	movdqa	%xmm0, (%rsp)
	movdqu	%xmm0, 28(%rsp)
	movdqa	%xmm0, 16(%rsp)
	movabsq	$4294967296, %rax       # imm = 0x100000000
.Ltmp12:
	#DEBUG_VALUE: fib:i <- undef
	#DEBUG_VALUE: fib:i <- [DW_OP_plus_uconst 1, DW_OP_stack_value] undef
	.loc	1 29 14                 # fib.c:29:14
	movq	%rax, (%rsp)
	.loc	1 31 14                 # fib.c:31:14
	movl	$1, 8(%rsp)
	movq	%rsp, %rsi
	.loc	1 33 10                 # fib.c:33:10
	callq	real_fib
.Ltmp13:
	.loc	1 0 10 is_stmt 0        # fib.c:0:10
	addq	$56, %rsp
	.cfi_def_cfa_offset 8
.LBB1_2:                                # %cleanup
	.loc	1 34 1 is_stmt 1        # fib.c:34:1
	retq
.Ltmp14:
.Lfunc_end1:
	.size	fib, .Lfunc_end1-fib
	.cfi_endproc
                                        # -- End function
	.globl	main                    # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin2:
	.loc	1 37 0                  # fib.c:37:0
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: main:argc <- $edi
	#DEBUG_VALUE: main:argv <- $rsi
	#DEBUG_VALUE: fib:x <- 3
	pushq	%r14
	.cfi_def_cfa_offset 16
	pushq	%rbx
	.cfi_def_cfa_offset 24
	subq	$56, %rsp
	.cfi_def_cfa_offset 80
	.cfi_offset %rbx, -24
	.cfi_offset %r14, -16
	.loc	1 27 16 prologue_end    # fib.c:27:16
	pcmpeqd	%xmm0, %xmm0
	movdqa	%xmm0, (%rsp)
	movdqu	%xmm0, 28(%rsp)
	movdqa	%xmm0, 16(%rsp)
	movabsq	$4294967296, %r14       # imm = 0x100000000
.Ltmp15:
	#DEBUG_VALUE: fib:i <- undef
	#DEBUG_VALUE: fib:i <- [DW_OP_plus_uconst 1, DW_OP_stack_value] undef
	.loc	1 29 14                 # fib.c:29:14
	movq	%r14, (%rsp)
	.loc	1 31 14                 # fib.c:31:14
	movl	$1, 8(%rsp)
	movq	%rsp, %rbx
	.loc	1 33 10                 # fib.c:33:10
	movl	$3, %edi
.Ltmp16:
	movq	%rbx, %rsi
.Ltmp17:
	callq	real_fib
.Ltmp18:
	#DEBUG_VALUE: main:result <- $eax
	.loc	1 41 3                  # fib.c:41:3
	movl	$.L.str, %edi
	movl	%eax, %esi
	xorl	%eax, %eax
.Ltmp19:
	callq	printf
.Ltmp20:
	.loc	1 27 16                 # fib.c:27:16
	pcmpeqd	%xmm0, %xmm0
.Ltmp21:
	#DEBUG_VALUE: fib:x <- 4
	.loc	1 27 16 is_stmt 0       # fib.c:27:16
	movdqa	%xmm0, (%rsp)
	movdqu	%xmm0, 28(%rsp)
	movdqa	%xmm0, 16(%rsp)
.Ltmp22:
	#DEBUG_VALUE: fib:i <- undef
	#DEBUG_VALUE: fib:i <- [DW_OP_plus_uconst 1, DW_OP_stack_value] undef
	.loc	1 29 14 is_stmt 1       # fib.c:29:14
	movq	%r14, (%rsp)
	.loc	1 31 14                 # fib.c:31:14
	movl	$1, 8(%rsp)
	.loc	1 33 10                 # fib.c:33:10
	movl	$4, %edi
	movq	%rbx, %rsi
	callq	real_fib
.Ltmp23:
	#DEBUG_VALUE: main:result <- $eax
	.loc	1 43 3                  # fib.c:43:3
	movl	$.L.str.1, %edi
	movl	%eax, %esi
	xorl	%eax, %eax
.Ltmp24:
	callq	printf
.Ltmp25:
	.loc	1 27 16                 # fib.c:27:16
	pcmpeqd	%xmm0, %xmm0
.Ltmp26:
	#DEBUG_VALUE: fib:x <- 5
	.loc	1 27 16 is_stmt 0       # fib.c:27:16
	movdqa	%xmm0, (%rsp)
	movdqu	%xmm0, 28(%rsp)
	movdqa	%xmm0, 16(%rsp)
.Ltmp27:
	#DEBUG_VALUE: fib:i <- undef
	#DEBUG_VALUE: fib:i <- [DW_OP_plus_uconst 1, DW_OP_stack_value] undef
	.loc	1 29 14 is_stmt 1       # fib.c:29:14
	movq	%r14, (%rsp)
	.loc	1 31 14                 # fib.c:31:14
	movl	$1, 8(%rsp)
	.loc	1 33 10                 # fib.c:33:10
	movl	$5, %edi
	movq	%rbx, %rsi
	callq	real_fib
.Ltmp28:
	#DEBUG_VALUE: main:result <- $eax
	.loc	1 45 3                  # fib.c:45:3
	movl	$.L.str.2, %edi
	movl	%eax, %esi
	xorl	%eax, %eax
.Ltmp29:
	callq	printf
.Ltmp30:
	.loc	1 27 16                 # fib.c:27:16
	pcmpeqd	%xmm0, %xmm0
.Ltmp31:
	#DEBUG_VALUE: fib:x <- 6
	.loc	1 27 16 is_stmt 0       # fib.c:27:16
	movdqa	%xmm0, (%rsp)
	movdqu	%xmm0, 28(%rsp)
	movdqa	%xmm0, 16(%rsp)
.Ltmp32:
	#DEBUG_VALUE: fib:i <- undef
	#DEBUG_VALUE: fib:i <- [DW_OP_plus_uconst 1, DW_OP_stack_value] undef
	.loc	1 29 14 is_stmt 1       # fib.c:29:14
	movq	%r14, (%rsp)
	.loc	1 31 14                 # fib.c:31:14
	movl	$1, 8(%rsp)
	.loc	1 33 10                 # fib.c:33:10
	movl	$6, %edi
	movq	%rbx, %rsi
	callq	real_fib
.Ltmp33:
	#DEBUG_VALUE: main:result <- $eax
	.loc	1 47 3                  # fib.c:47:3
	movl	$.L.str.3, %edi
	movl	%eax, %esi
	xorl	%eax, %eax
.Ltmp34:
	callq	printf
.Ltmp35:
	.loc	1 27 16                 # fib.c:27:16
	pcmpeqd	%xmm0, %xmm0
.Ltmp36:
	#DEBUG_VALUE: fib:x <- 7
	.loc	1 27 16 is_stmt 0       # fib.c:27:16
	movdqa	%xmm0, (%rsp)
	movdqu	%xmm0, 28(%rsp)
	movdqa	%xmm0, 16(%rsp)
.Ltmp37:
	#DEBUG_VALUE: fib:i <- undef
	#DEBUG_VALUE: fib:i <- [DW_OP_plus_uconst 1, DW_OP_stack_value] undef
	.loc	1 29 14 is_stmt 1       # fib.c:29:14
	movq	%r14, (%rsp)
	.loc	1 31 14                 # fib.c:31:14
	movl	$1, 8(%rsp)
	.loc	1 33 10                 # fib.c:33:10
	movl	$7, %edi
	movq	%rbx, %rsi
	callq	real_fib
.Ltmp38:
	#DEBUG_VALUE: main:result <- $eax
	.loc	1 49 3                  # fib.c:49:3
	movl	$.L.str.4, %edi
	movl	%eax, %esi
	xorl	%eax, %eax
.Ltmp39:
	callq	printf
.Ltmp40:
	.loc	1 27 16                 # fib.c:27:16
	pcmpeqd	%xmm0, %xmm0
.Ltmp41:
	#DEBUG_VALUE: fib:x <- 8
	.loc	1 27 16 is_stmt 0       # fib.c:27:16
	movdqa	%xmm0, (%rsp)
	movdqu	%xmm0, 28(%rsp)
	movdqa	%xmm0, 16(%rsp)
.Ltmp42:
	#DEBUG_VALUE: fib:i <- undef
	#DEBUG_VALUE: fib:i <- [DW_OP_plus_uconst 1, DW_OP_stack_value] undef
	.loc	1 29 14 is_stmt 1       # fib.c:29:14
	movq	%r14, (%rsp)
	.loc	1 31 14                 # fib.c:31:14
	movl	$1, 8(%rsp)
	.loc	1 33 10                 # fib.c:33:10
	movl	$8, %edi
	movq	%rbx, %rsi
	callq	real_fib
.Ltmp43:
	#DEBUG_VALUE: main:result <- $eax
	.loc	1 51 3                  # fib.c:51:3
	movl	$.L.str.5, %edi
	movl	%eax, %esi
	xorl	%eax, %eax
.Ltmp44:
	callq	printf
.Ltmp45:
	.loc	1 27 16                 # fib.c:27:16
	pcmpeqd	%xmm0, %xmm0
.Ltmp46:
	#DEBUG_VALUE: fib:x <- 9
	.loc	1 27 16 is_stmt 0       # fib.c:27:16
	movdqa	%xmm0, (%rsp)
	movdqu	%xmm0, 28(%rsp)
	movdqa	%xmm0, 16(%rsp)
.Ltmp47:
	#DEBUG_VALUE: fib:i <- undef
	#DEBUG_VALUE: fib:i <- [DW_OP_plus_uconst 1, DW_OP_stack_value] undef
	.loc	1 29 14 is_stmt 1       # fib.c:29:14
	movq	%r14, (%rsp)
	.loc	1 31 14                 # fib.c:31:14
	movl	$1, 8(%rsp)
	.loc	1 33 10                 # fib.c:33:10
	movl	$9, %edi
	movq	%rbx, %rsi
	callq	real_fib
.Ltmp48:
	#DEBUG_VALUE: main:result <- $eax
	.loc	1 53 3                  # fib.c:53:3
	movl	$.L.str.6, %edi
	movl	%eax, %esi
	xorl	%eax, %eax
.Ltmp49:
	callq	printf
.Ltmp50:
	.loc	1 27 16                 # fib.c:27:16
	pcmpeqd	%xmm0, %xmm0
.Ltmp51:
	#DEBUG_VALUE: fib:x <- 10
	.loc	1 27 16 is_stmt 0       # fib.c:27:16
	movdqa	%xmm0, (%rsp)
	movdqu	%xmm0, 28(%rsp)
	movdqa	%xmm0, 16(%rsp)
.Ltmp52:
	#DEBUG_VALUE: fib:i <- undef
	#DEBUG_VALUE: fib:i <- [DW_OP_plus_uconst 1, DW_OP_stack_value] undef
	.loc	1 29 14 is_stmt 1       # fib.c:29:14
	movq	%r14, (%rsp)
	.loc	1 31 14                 # fib.c:31:14
	movl	$1, 8(%rsp)
	.loc	1 33 10                 # fib.c:33:10
	movl	$10, %edi
	movq	%rbx, %rsi
	callq	real_fib
.Ltmp53:
	#DEBUG_VALUE: main:result <- $eax
	.loc	1 55 3                  # fib.c:55:3
	movl	$.L.str.7, %edi
	movl	%eax, %esi
	xorl	%eax, %eax
.Ltmp54:
	callq	printf
	.loc	1 57 3                  # fib.c:57:3
	xorl	%eax, %eax
	addq	$56, %rsp
	.cfi_def_cfa_offset 24
	popq	%rbx
	.cfi_def_cfa_offset 16
	popq	%r14
	.cfi_def_cfa_offset 8
	retq
.Ltmp55:
.Lfunc_end2:
	.size	main, .Lfunc_end2-main
	.cfi_endproc
                                        # -- End function
	.type	.L.str,@object          # @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"fibonacci(3) = %d\n"
	.size	.L.str, 19

	.type	.L.str.1,@object        # @.str.1
.L.str.1:
	.asciz	"fibonacci(4) = %d\n"
	.size	.L.str.1, 19

	.type	.L.str.2,@object        # @.str.2
.L.str.2:
	.asciz	"fibonacci(5) = %d\n"
	.size	.L.str.2, 19

	.type	.L.str.3,@object        # @.str.3
.L.str.3:
	.asciz	"fibonacci(6) = %d\n"
	.size	.L.str.3, 19

	.type	.L.str.4,@object        # @.str.4
.L.str.4:
	.asciz	"fibonacci(7) = %d\n"
	.size	.L.str.4, 19

	.type	.L.str.5,@object        # @.str.5
.L.str.5:
	.asciz	"fibonacci(8) = %d\n"
	.size	.L.str.5, 19

	.type	.L.str.6,@object        # @.str.6
.L.str.6:
	.asciz	"fibonacci(9) = %d\n"
	.size	.L.str.6, 19

	.type	.L.str.7,@object        # @.str.7
.L.str.7:
	.asciz	"fibonacci(10) = %d\n"
	.size	.L.str.7, 20

	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 9.0.0 (trunk 355041)" # string offset=0
.Linfo_string1:
	.asciz	"fib.c"                 # string offset=35
.Linfo_string2:
	.asciz	"/usr/local/google/home/cmtice/c++-tests" # string offset=41
.Linfo_string3:
	.asciz	"fib"                   # string offset=81
.Linfo_string4:
	.asciz	"int"                   # string offset=85
.Linfo_string5:
	.asciz	"x"                     # string offset=89
.Linfo_string6:
	.asciz	"answers"               # string offset=91
.Linfo_string7:
	.asciz	"__ARRAY_SIZE_TYPE__"   # string offset=99
.Linfo_string8:
	.asciz	"i"                     # string offset=119
.Linfo_string9:
	.asciz	"real_fib"              # string offset=121
.Linfo_string10:
	.asciz	"main"                  # string offset=130
.Linfo_string11:
	.asciz	"result"                # string offset=135
.Linfo_string12:
	.asciz	"argc"                  # string offset=142
.Linfo_string13:
	.asciz	"argv"                  # string offset=147
.Linfo_string14:
	.asciz	"char"                  # string offset=152
	.section	.debug_loc,"",@progbits
.Ldebug_loc0:
	.quad	.Lfunc_begin0-.Lfunc_begin0
	.quad	.Ltmp4-.Lfunc_begin0
	.short	1                       # Loc expr size
	.byte	85                      # super-register DW_OP_reg5
	.quad	0
	.quad	0
.Ldebug_loc1:
	.quad	.Lfunc_begin0-.Lfunc_begin0
	.quad	.Ltmp3-.Lfunc_begin0
	.short	1                       # Loc expr size
	.byte	84                      # DW_OP_reg4
	.quad	.Ltmp3-.Lfunc_begin0
	.quad	.Ltmp6-.Lfunc_begin0
	.short	1                       # Loc expr size
	.byte	83                      # DW_OP_reg3
	.quad	0
	.quad	0
.Ldebug_loc2:
	.quad	.Ltmp5-.Lfunc_begin0
	.quad	.Lfunc_end0-.Lfunc_begin0
	.short	1                       # Loc expr size
	.byte	80                      # super-register DW_OP_reg0
	.quad	0
	.quad	0
.Ldebug_loc3:
	.quad	.Lfunc_begin1-.Lfunc_begin0
	.quad	.Ltmp13-.Lfunc_begin0
	.short	1                       # Loc expr size
	.byte	85                      # super-register DW_OP_reg5
	.quad	0
	.quad	0
.Ldebug_loc4:
	.quad	.Lfunc_begin2-.Lfunc_begin0
	.quad	.Ltmp16-.Lfunc_begin0
	.short	1                       # Loc expr size
	.byte	85                      # super-register DW_OP_reg5
	.quad	0
	.quad	0
.Ldebug_loc5:
	.quad	.Lfunc_begin2-.Lfunc_begin0
	.quad	.Ltmp17-.Lfunc_begin0
	.short	1                       # Loc expr size
	.byte	84                      # DW_OP_reg4
	.quad	0
	.quad	0
.Ldebug_loc6:
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp19-.Lfunc_begin0
	.short	1                       # Loc expr size
	.byte	80                      # super-register DW_OP_reg0
	.quad	.Ltmp23-.Lfunc_begin0
	.quad	.Ltmp24-.Lfunc_begin0
	.short	1                       # Loc expr size
	.byte	80                      # super-register DW_OP_reg0
	.quad	.Ltmp28-.Lfunc_begin0
	.quad	.Ltmp29-.Lfunc_begin0
	.short	1                       # Loc expr size
	.byte	80                      # super-register DW_OP_reg0
	.quad	.Ltmp33-.Lfunc_begin0
	.quad	.Ltmp34-.Lfunc_begin0
	.short	1                       # Loc expr size
	.byte	80                      # super-register DW_OP_reg0
	.quad	.Ltmp38-.Lfunc_begin0
	.quad	.Ltmp39-.Lfunc_begin0
	.short	1                       # Loc expr size
	.byte	80                      # super-register DW_OP_reg0
	.quad	.Ltmp43-.Lfunc_begin0
	.quad	.Ltmp44-.Lfunc_begin0
	.short	1                       # Loc expr size
	.byte	80                      # super-register DW_OP_reg0
	.quad	.Ltmp48-.Lfunc_begin0
	.quad	.Ltmp49-.Lfunc_begin0
	.short	1                       # Loc expr size
	.byte	80                      # super-register DW_OP_reg0
	.quad	.Ltmp53-.Lfunc_begin0
	.quad	.Ltmp54-.Lfunc_begin0
	.short	1                       # Loc expr size
	.byte	80                      # super-register DW_OP_reg0
	.quad	0
	.quad	0
.Ldebug_loc7:
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Lfunc_end2-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	17                      # DW_OP_consts
	.byte	4                       # 4
	.byte	159                     # DW_OP_stack_value
	.quad	0
	.quad	0
.Ldebug_loc8:
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Lfunc_end2-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	17                      # DW_OP_consts
	.byte	5                       # 5
	.byte	159                     # DW_OP_stack_value
	.quad	0
	.quad	0
.Ldebug_loc9:
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Lfunc_end2-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	17                      # DW_OP_consts
	.byte	6                       # 6
	.byte	159                     # DW_OP_stack_value
	.quad	0
	.quad	0
.Ldebug_loc10:
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Lfunc_end2-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	17                      # DW_OP_consts
	.byte	7                       # 7
	.byte	159                     # DW_OP_stack_value
	.quad	0
	.quad	0
.Ldebug_loc11:
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Lfunc_end2-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	17                      # DW_OP_consts
	.byte	8                       # 8
	.byte	159                     # DW_OP_stack_value
	.quad	0
	.quad	0
.Ldebug_loc12:
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Lfunc_end2-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	17                      # DW_OP_consts
	.byte	9                       # 9
	.byte	159                     # DW_OP_stack_value
	.quad	0
	.quad	0
.Ldebug_loc13:
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	.Lfunc_end2-.Lfunc_begin0
	.short	3                       # Loc expr size
	.byte	17                      # DW_OP_consts
	.byte	10                      # 10
	.byte	159                     # DW_OP_stack_value
	.quad	0
	.quad	0
	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	37                      # DW_AT_producer
	.byte	14                      # DW_FORM_strp
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	16                      # DW_AT_stmt_list
	.byte	23                      # DW_FORM_sec_offset
	.byte	27                      # DW_AT_comp_dir
	.byte	14                      # DW_FORM_strp
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	2                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	1                       # DW_CHILDREN_yes
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	64                      # DW_AT_frame_base
	.byte	24                      # DW_FORM_exprloc
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	39                      # DW_AT_prototyped
	.byte	25                      # DW_FORM_flag_present
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	63                      # DW_AT_external
	.byte	25                      # DW_FORM_flag_present
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	3                       # Abbreviation Code
	.byte	5                       # DW_TAG_formal_parameter
	.byte	0                       # DW_CHILDREN_no
	.byte	2                       # DW_AT_location
	.byte	23                      # DW_FORM_sec_offset
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	4                       # Abbreviation Code
	.byte	52                      # DW_TAG_variable
	.byte	0                       # DW_CHILDREN_no
	.byte	2                       # DW_AT_location
	.byte	23                      # DW_FORM_sec_offset
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	5                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	1                       # DW_CHILDREN_yes
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	64                      # DW_AT_frame_base
	.byte	24                      # DW_FORM_exprloc
	.byte	49                      # DW_AT_abstract_origin
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	6                       # Abbreviation Code
	.byte	5                       # DW_TAG_formal_parameter
	.byte	0                       # DW_CHILDREN_no
	.byte	2                       # DW_AT_location
	.byte	23                      # DW_FORM_sec_offset
	.byte	49                      # DW_AT_abstract_origin
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	7                       # Abbreviation Code
	.byte	52                      # DW_TAG_variable
	.byte	0                       # DW_CHILDREN_no
	.byte	2                       # DW_AT_location
	.byte	24                      # DW_FORM_exprloc
	.byte	49                      # DW_AT_abstract_origin
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	8                       # Abbreviation Code
	.byte	52                      # DW_TAG_variable
	.byte	0                       # DW_CHILDREN_no
	.byte	49                      # DW_AT_abstract_origin
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	9                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	1                       # DW_CHILDREN_yes
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	39                      # DW_AT_prototyped
	.byte	25                      # DW_FORM_flag_present
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	63                      # DW_AT_external
	.byte	25                      # DW_FORM_flag_present
	.byte	32                      # DW_AT_inline
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	10                      # Abbreviation Code
	.byte	5                       # DW_TAG_formal_parameter
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	11                      # Abbreviation Code
	.byte	52                      # DW_TAG_variable
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	12                      # Abbreviation Code
	.byte	36                      # DW_TAG_base_type
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	62                      # DW_AT_encoding
	.byte	11                      # DW_FORM_data1
	.byte	11                      # DW_AT_byte_size
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	13                      # Abbreviation Code
	.byte	1                       # DW_TAG_array_type
	.byte	1                       # DW_CHILDREN_yes
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	14                      # Abbreviation Code
	.byte	33                      # DW_TAG_subrange_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	55                      # DW_AT_count
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	15                      # Abbreviation Code
	.byte	36                      # DW_TAG_base_type
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	11                      # DW_AT_byte_size
	.byte	11                      # DW_FORM_data1
	.byte	62                      # DW_AT_encoding
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	16                      # Abbreviation Code
	.byte	29                      # DW_TAG_inlined_subroutine
	.byte	1                       # DW_CHILDREN_yes
	.byte	49                      # DW_AT_abstract_origin
	.byte	19                      # DW_FORM_ref4
	.byte	85                      # DW_AT_ranges
	.byte	23                      # DW_FORM_sec_offset
	.byte	88                      # DW_AT_call_file
	.byte	11                      # DW_FORM_data1
	.byte	89                      # DW_AT_call_line
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	17                      # Abbreviation Code
	.byte	5                       # DW_TAG_formal_parameter
	.byte	0                       # DW_CHILDREN_no
	.byte	28                      # DW_AT_const_value
	.byte	13                      # DW_FORM_sdata
	.byte	49                      # DW_AT_abstract_origin
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	18                      # Abbreviation Code
	.byte	29                      # DW_TAG_inlined_subroutine
	.byte	1                       # DW_CHILDREN_yes
	.byte	49                      # DW_AT_abstract_origin
	.byte	19                      # DW_FORM_ref4
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	88                      # DW_AT_call_file
	.byte	11                      # DW_FORM_data1
	.byte	89                      # DW_AT_call_line
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	19                      # Abbreviation Code
	.byte	15                      # DW_TAG_pointer_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                       # DWARF version number
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] 0xb:0x27b DW_TAG_compile_unit
	.long	.Linfo_string0          # DW_AT_producer
	.short	12                      # DW_AT_language
	.long	.Linfo_string1          # DW_AT_name
	.long	.Lline_table_start0     # DW_AT_stmt_list
	.long	.Linfo_string2          # DW_AT_comp_dir
	.quad	.Lfunc_begin0           # DW_AT_low_pc
	.long	.Lfunc_end2-.Lfunc_begin0 # DW_AT_high_pc
	.byte	2                       # Abbrev [2] 0x2a:0x47 DW_TAG_subprogram
	.quad	.Lfunc_begin0           # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
	.byte	1                       # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string9          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	4                       # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	201                     # DW_AT_type
                                        # DW_AT_external
	.byte	3                       # Abbrev [3] 0x43:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc0            # DW_AT_location
	.long	.Linfo_string5          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	4                       # DW_AT_decl_line
	.long	201                     # DW_AT_type
	.byte	3                       # Abbrev [3] 0x52:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc1            # DW_AT_location
	.long	.Linfo_string6          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	4                       # DW_AT_decl_line
	.long	623                     # DW_AT_type
	.byte	4                       # Abbrev [4] 0x61:0xf DW_TAG_variable
	.long	.Ldebug_loc2            # DW_AT_location
	.long	.Linfo_string11         # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	6                       # DW_AT_decl_line
	.long	201                     # DW_AT_type
	.byte	0                       # End Of Children Mark
	.byte	5                       # Abbrev [5] 0x71:0x2a DW_TAG_subprogram
	.quad	.Lfunc_begin1           # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1 # DW_AT_high_pc
	.byte	1                       # DW_AT_frame_base
	.byte	87
	.long	155                     # DW_AT_abstract_origin
	.byte	6                       # Abbrev [6] 0x84:0x9 DW_TAG_formal_parameter
	.long	.Ldebug_loc3            # DW_AT_location
	.long	167                     # DW_AT_abstract_origin
	.byte	7                       # Abbrev [7] 0x8d:0x8 DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	0
	.long	178                     # DW_AT_abstract_origin
	.byte	8                       # Abbrev [8] 0x95:0x5 DW_TAG_variable
	.long	189                     # DW_AT_abstract_origin
	.byte	0                       # End Of Children Mark
	.byte	9                       # Abbrev [9] 0x9b:0x2e DW_TAG_subprogram
	.long	.Linfo_string3          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	18                      # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	201                     # DW_AT_type
                                        # DW_AT_external
	.byte	1                       # DW_AT_inline
	.byte	10                      # Abbrev [10] 0xa7:0xb DW_TAG_formal_parameter
	.long	.Linfo_string5          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	18                      # DW_AT_decl_line
	.long	201                     # DW_AT_type
	.byte	11                      # Abbrev [11] 0xb2:0xb DW_TAG_variable
	.long	.Linfo_string6          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	20                      # DW_AT_decl_line
	.long	208                     # DW_AT_type
	.byte	11                      # Abbrev [11] 0xbd:0xb DW_TAG_variable
	.long	.Linfo_string8          # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	21                      # DW_AT_decl_line
	.long	201                     # DW_AT_type
	.byte	0                       # End Of Children Mark
	.byte	12                      # Abbrev [12] 0xc9:0x7 DW_TAG_base_type
	.long	.Linfo_string4          # DW_AT_name
	.byte	5                       # DW_AT_encoding
	.byte	4                       # DW_AT_byte_size
	.byte	13                      # Abbrev [13] 0xd0:0xc DW_TAG_array_type
	.long	201                     # DW_AT_type
	.byte	14                      # Abbrev [14] 0xd5:0x6 DW_TAG_subrange_type
	.long	220                     # DW_AT_type
	.byte	11                      # DW_AT_count
	.byte	0                       # End Of Children Mark
	.byte	15                      # Abbrev [15] 0xdc:0x7 DW_TAG_base_type
	.long	.Linfo_string7          # DW_AT_name
	.byte	8                       # DW_AT_byte_size
	.byte	7                       # DW_AT_encoding
	.byte	2                       # Abbrev [2] 0xe3:0x18c DW_TAG_subprogram
	.quad	.Lfunc_begin2           # DW_AT_low_pc
	.long	.Lfunc_end2-.Lfunc_begin2 # DW_AT_high_pc
	.byte	1                       # DW_AT_frame_base
	.byte	87
	.long	.Linfo_string10         # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	36                      # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	201                     # DW_AT_type
                                        # DW_AT_external
	.byte	3                       # Abbrev [3] 0xfc:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc4            # DW_AT_location
	.long	.Linfo_string12         # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	36                      # DW_AT_decl_line
	.long	201                     # DW_AT_type
	.byte	3                       # Abbrev [3] 0x10b:0xf DW_TAG_formal_parameter
	.long	.Ldebug_loc5            # DW_AT_location
	.long	.Linfo_string13         # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	36                      # DW_AT_decl_line
	.long	628                     # DW_AT_type
	.byte	4                       # Abbrev [4] 0x11a:0xf DW_TAG_variable
	.long	.Ldebug_loc6            # DW_AT_location
	.long	.Linfo_string11         # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	38                      # DW_AT_decl_line
	.long	201                     # DW_AT_type
	.byte	16                      # Abbrev [16] 0x129:0x1f DW_TAG_inlined_subroutine
	.long	155                     # DW_AT_abstract_origin
	.long	.Ldebug_ranges0         # DW_AT_ranges
	.byte	1                       # DW_AT_call_file
	.byte	40                      # DW_AT_call_line
	.byte	17                      # Abbrev [17] 0x134:0x6 DW_TAG_formal_parameter
	.byte	3                       # DW_AT_const_value
	.long	167                     # DW_AT_abstract_origin
	.byte	7                       # Abbrev [7] 0x13a:0x8 DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	0
	.long	178                     # DW_AT_abstract_origin
	.byte	8                       # Abbrev [8] 0x142:0x5 DW_TAG_variable
	.long	189                     # DW_AT_abstract_origin
	.byte	0                       # End Of Children Mark
	.byte	18                      # Abbrev [18] 0x148:0x2a DW_TAG_inlined_subroutine
	.long	155                     # DW_AT_abstract_origin
	.quad	.Ltmp21                 # DW_AT_low_pc
	.long	.Ltmp23-.Ltmp21         # DW_AT_high_pc
	.byte	1                       # DW_AT_call_file
	.byte	42                      # DW_AT_call_line
	.byte	6                       # Abbrev [6] 0x15b:0x9 DW_TAG_formal_parameter
	.long	.Ldebug_loc7            # DW_AT_location
	.long	167                     # DW_AT_abstract_origin
	.byte	7                       # Abbrev [7] 0x164:0x8 DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	0
	.long	178                     # DW_AT_abstract_origin
	.byte	8                       # Abbrev [8] 0x16c:0x5 DW_TAG_variable
	.long	189                     # DW_AT_abstract_origin
	.byte	0                       # End Of Children Mark
	.byte	18                      # Abbrev [18] 0x172:0x2a DW_TAG_inlined_subroutine
	.long	155                     # DW_AT_abstract_origin
	.quad	.Ltmp26                 # DW_AT_low_pc
	.long	.Ltmp28-.Ltmp26         # DW_AT_high_pc
	.byte	1                       # DW_AT_call_file
	.byte	44                      # DW_AT_call_line
	.byte	6                       # Abbrev [6] 0x185:0x9 DW_TAG_formal_parameter
	.long	.Ldebug_loc8            # DW_AT_location
	.long	167                     # DW_AT_abstract_origin
	.byte	7                       # Abbrev [7] 0x18e:0x8 DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	0
	.long	178                     # DW_AT_abstract_origin
	.byte	8                       # Abbrev [8] 0x196:0x5 DW_TAG_variable
	.long	189                     # DW_AT_abstract_origin
	.byte	0                       # End Of Children Mark
	.byte	18                      # Abbrev [18] 0x19c:0x2a DW_TAG_inlined_subroutine
	.long	155                     # DW_AT_abstract_origin
	.quad	.Ltmp31                 # DW_AT_low_pc
	.long	.Ltmp33-.Ltmp31         # DW_AT_high_pc
	.byte	1                       # DW_AT_call_file
	.byte	46                      # DW_AT_call_line
	.byte	6                       # Abbrev [6] 0x1af:0x9 DW_TAG_formal_parameter
	.long	.Ldebug_loc9            # DW_AT_location
	.long	167                     # DW_AT_abstract_origin
	.byte	7                       # Abbrev [7] 0x1b8:0x8 DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	0
	.long	178                     # DW_AT_abstract_origin
	.byte	8                       # Abbrev [8] 0x1c0:0x5 DW_TAG_variable
	.long	189                     # DW_AT_abstract_origin
	.byte	0                       # End Of Children Mark
	.byte	18                      # Abbrev [18] 0x1c6:0x2a DW_TAG_inlined_subroutine
	.long	155                     # DW_AT_abstract_origin
	.quad	.Ltmp36                 # DW_AT_low_pc
	.long	.Ltmp38-.Ltmp36         # DW_AT_high_pc
	.byte	1                       # DW_AT_call_file
	.byte	48                      # DW_AT_call_line
	.byte	6                       # Abbrev [6] 0x1d9:0x9 DW_TAG_formal_parameter
	.long	.Ldebug_loc10           # DW_AT_location
	.long	167                     # DW_AT_abstract_origin
	.byte	7                       # Abbrev [7] 0x1e2:0x8 DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	0
	.long	178                     # DW_AT_abstract_origin
	.byte	8                       # Abbrev [8] 0x1ea:0x5 DW_TAG_variable
	.long	189                     # DW_AT_abstract_origin
	.byte	0                       # End Of Children Mark
	.byte	18                      # Abbrev [18] 0x1f0:0x2a DW_TAG_inlined_subroutine
	.long	155                     # DW_AT_abstract_origin
	.quad	.Ltmp41                 # DW_AT_low_pc
	.long	.Ltmp43-.Ltmp41         # DW_AT_high_pc
	.byte	1                       # DW_AT_call_file
	.byte	50                      # DW_AT_call_line
	.byte	6                       # Abbrev [6] 0x203:0x9 DW_TAG_formal_parameter
	.long	.Ldebug_loc11           # DW_AT_location
	.long	167                     # DW_AT_abstract_origin
	.byte	7                       # Abbrev [7] 0x20c:0x8 DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	0
	.long	178                     # DW_AT_abstract_origin
	.byte	8                       # Abbrev [8] 0x214:0x5 DW_TAG_variable
	.long	189                     # DW_AT_abstract_origin
	.byte	0                       # End Of Children Mark
	.byte	18                      # Abbrev [18] 0x21a:0x2a DW_TAG_inlined_subroutine
	.long	155                     # DW_AT_abstract_origin
	.quad	.Ltmp46                 # DW_AT_low_pc
	.long	.Ltmp48-.Ltmp46         # DW_AT_high_pc
	.byte	1                       # DW_AT_call_file
	.byte	52                      # DW_AT_call_line
	.byte	6                       # Abbrev [6] 0x22d:0x9 DW_TAG_formal_parameter
	.long	.Ldebug_loc12           # DW_AT_location
	.long	167                     # DW_AT_abstract_origin
	.byte	7                       # Abbrev [7] 0x236:0x8 DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	0
	.long	178                     # DW_AT_abstract_origin
	.byte	8                       # Abbrev [8] 0x23e:0x5 DW_TAG_variable
	.long	189                     # DW_AT_abstract_origin
	.byte	0                       # End Of Children Mark
	.byte	18                      # Abbrev [18] 0x244:0x2a DW_TAG_inlined_subroutine
	.long	155                     # DW_AT_abstract_origin
	.quad	.Ltmp51                 # DW_AT_low_pc
	.long	.Ltmp53-.Ltmp51         # DW_AT_high_pc
	.byte	1                       # DW_AT_call_file
	.byte	54                      # DW_AT_call_line
	.byte	6                       # Abbrev [6] 0x257:0x9 DW_TAG_formal_parameter
	.long	.Ldebug_loc13           # DW_AT_location
	.long	167                     # DW_AT_abstract_origin
	.byte	7                       # Abbrev [7] 0x260:0x8 DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	0
	.long	178                     # DW_AT_abstract_origin
	.byte	8                       # Abbrev [8] 0x268:0x5 DW_TAG_variable
	.long	189                     # DW_AT_abstract_origin
	.byte	0                       # End Of Children Mark
	.byte	0                       # End Of Children Mark
	.byte	19                      # Abbrev [19] 0x26f:0x5 DW_TAG_pointer_type
	.long	201                     # DW_AT_type
	.byte	19                      # Abbrev [19] 0x274:0x5 DW_TAG_pointer_type
	.long	633                     # DW_AT_type
	.byte	19                      # Abbrev [19] 0x279:0x5 DW_TAG_pointer_type
	.long	638                     # DW_AT_type
	.byte	12                      # Abbrev [12] 0x27e:0x7 DW_TAG_base_type
	.long	.Linfo_string14         # DW_AT_name
	.byte	6                       # DW_AT_encoding
	.byte	1                       # DW_AT_byte_size
	.byte	0                       # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.quad	.Lfunc_begin2-.Lfunc_begin0
	.quad	.Ltmp18-.Lfunc_begin0
	.quad	.Ltmp20-.Lfunc_begin0
	.quad	.Ltmp21-.Lfunc_begin0
	.quad	.Ltmp25-.Lfunc_begin0
	.quad	.Ltmp26-.Lfunc_begin0
	.quad	.Ltmp30-.Lfunc_begin0
	.quad	.Ltmp31-.Lfunc_begin0
	.quad	.Ltmp35-.Lfunc_begin0
	.quad	.Ltmp36-.Lfunc_begin0
	.quad	.Ltmp40-.Lfunc_begin0
	.quad	.Ltmp41-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	.Ltmp46-.Lfunc_begin0
	.quad	.Ltmp50-.Lfunc_begin0
	.quad	.Ltmp51-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_macinfo,"",@progbits
	.byte	0                       # End Of Macro List Mark

	.ident	"clang version 9.0.0 (trunk 355041)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
