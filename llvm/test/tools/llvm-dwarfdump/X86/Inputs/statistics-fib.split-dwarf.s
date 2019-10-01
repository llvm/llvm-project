	.text
	.file	"fibonacci.cc"
	.globl	_Z8real_fibiPi          # -- Begin function _Z8real_fibiPi
	.p2align	4, 0x90
	.type	_Z8real_fibiPi,@function
_Z8real_fibiPi:                         # @_Z8real_fibiPi
.Lfunc_begin0:
	.file	1 "/usr/local/google3/cmtice/llvm.tot2/build/test/tools/llvm-dwarfdump/X86/Output/statistics-dwo.test.tmp" "fibonacci.cc"
	.loc	1 5 0                   # fibonacci.cc:5:0
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
	.loc	1 8 7 prologue_end      # fibonacci.cc:8:7
	movslq	%edi, %rbp
	movl	(%rsi,%rbp,4), %eax
	.loc	1 8 20 is_stmt 0        # fibonacci.cc:8:20
	cmpl	$-1, %eax
.Ltmp0:
	.loc	1 8 7                   # fibonacci.cc:8:7
	je	.LBB0_1
.Ltmp1:
# %bb.2:                                # %cleanup
	#DEBUG_VALUE: real_fib:answers <- $rsi
	#DEBUG_VALUE: real_fib:x <- $edi
	.loc	1 15 1 is_stmt 1        # fibonacci.cc:15:1
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
	.loc	1 0 1 is_stmt 0         # fibonacci.cc:0:1
	movq	%rsi, %rbx
.Ltmp3:
	#DEBUG_VALUE: real_fib:answers <- $rbx
	.loc	1 11 22 is_stmt 1       # fibonacci.cc:11:22
	leal	-1(%rbp), %edi
.Ltmp4:
	.loc	1 11 12 is_stmt 0       # fibonacci.cc:11:12
	callq	_Z8real_fibiPi
	movl	%eax, %r14d
	.loc	1 11 47                 # fibonacci.cc:11:47
	leal	-2(%rbp), %edi
	.loc	1 11 37                 # fibonacci.cc:11:37
	movq	%rbx, %rsi
	callq	_Z8real_fibiPi
	.loc	1 11 35                 # fibonacci.cc:11:35
	addl	%r14d, %eax
.Ltmp5:
	#DEBUG_VALUE: real_fib:result <- $eax
	.loc	1 12 16 is_stmt 1       # fibonacci.cc:12:16
	movl	%eax, (%rbx,%rbp,4)
	.loc	1 15 1                  # fibonacci.cc:15:1
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
	.size	_Z8real_fibiPi, .Lfunc_end0-_Z8real_fibiPi
	.cfi_endproc
                                        # -- End function
	.globl	_Z3fibi                 # -- Begin function _Z3fibi
	.p2align	4, 0x90
	.type	_Z3fibi,@function
_Z3fibi:                                # @_Z3fibi
.Lfunc_begin1:
	.loc	1 19 0                  # fibonacci.cc:19:0
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: fib:x <- $edi
	movl	$-1, %eax
	#DEBUG_VALUE: fib:x <- $edi
.Ltmp8:
	.loc	1 23 9 prologue_end     # fibonacci.cc:23:9
	cmpl	$10, %edi
.Ltmp9:
	.loc	1 23 7 is_stmt 0        # fibonacci.cc:23:7
	jg	.LBB1_2
.Ltmp10:
# %bb.1:                                # %for.body.preheader
	#DEBUG_VALUE: fib:x <- $edi
	subq	$56, %rsp
	.cfi_def_cfa_offset 64
.Ltmp11:
	.loc	1 27 16 is_stmt 1       # fibonacci.cc:27:16
	pcmpeqd	%xmm0, %xmm0
	movdqa	%xmm0, (%rsp)
	movdqu	%xmm0, 28(%rsp)
	movdqa	%xmm0, 16(%rsp)
	movabsq	$4294967296, %rax       # imm = 0x100000000
.Ltmp12:
	#DEBUG_VALUE: fib:i <- undef
	#DEBUG_VALUE: fib:i <- [DW_OP_plus_uconst 1, DW_OP_stack_value] undef
	.loc	1 29 14                 # fibonacci.cc:29:14
	movq	%rax, (%rsp)
	.loc	1 31 14                 # fibonacci.cc:31:14
	movl	$1, 8(%rsp)
	movq	%rsp, %rsi
	.loc	1 33 10                 # fibonacci.cc:33:10
	callq	_Z8real_fibiPi
.Ltmp13:
	.loc	1 0 10 is_stmt 0        # fibonacci.cc:0:10
	addq	$56, %rsp
	.cfi_def_cfa_offset 8
.LBB1_2:                                # %cleanup
	.loc	1 34 1 is_stmt 1        # fibonacci.cc:34:1
	retq
.Ltmp14:
.Lfunc_end1:
	.size	_Z3fibi, .Lfunc_end1-_Z3fibi
	.cfi_endproc
                                        # -- End function
	.globl	main                    # -- Begin function main
	.p2align	4, 0x90
	.type	main,@function
main:                                   # @main
.Lfunc_begin2:
	.loc	1 37 0                  # fibonacci.cc:37:0
	.cfi_startproc
# %bb.0:                                # %entry
	#DEBUG_VALUE: main:argc <- $edi
	#DEBUG_VALUE: main:argv <- $rsi
	#DEBUG_VALUE: fib:x <- 3
	pushq	%rbp
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%rbx
	.cfi_def_cfa_offset 32
	subq	$48, %rsp
	.cfi_def_cfa_offset 80
	.cfi_offset %rbx, -32
	.cfi_offset %r14, -24
	.cfi_offset %rbp, -16
	.loc	1 27 16 prologue_end    # fibonacci.cc:27:16
	pcmpeqd	%xmm0, %xmm0
	movdqa	%xmm0, (%rsp)
	movdqu	%xmm0, 28(%rsp)
	movdqa	%xmm0, 16(%rsp)
	movabsq	$4294967296, %r14       # imm = 0x100000000
.Ltmp15:
	#DEBUG_VALUE: fib:i <- undef
	#DEBUG_VALUE: fib:i <- [DW_OP_plus_uconst 1, DW_OP_stack_value] undef
	.loc	1 29 14                 # fibonacci.cc:29:14
	movq	%r14, (%rsp)
	.loc	1 31 14                 # fibonacci.cc:31:14
	movl	$1, 8(%rsp)
	movq	%rsp, %rbx
	.loc	1 33 10                 # fibonacci.cc:33:10
	movl	$3, %edi
.Ltmp16:
	movq	%rbx, %rsi
.Ltmp17:
	callq	_Z8real_fibiPi
.Ltmp18:
	#DEBUG_VALUE: main:result <- $eax
	.loc	1 41 3                  # fibonacci.cc:41:3
	movl	$.L.str, %edi
	movl	%eax, %esi
	xorl	%eax, %eax
.Ltmp19:
	callq	printf
.Ltmp20:
	.loc	1 27 16                 # fibonacci.cc:27:16
	pcmpeqd	%xmm0, %xmm0
.Ltmp21:
	#DEBUG_VALUE: fib:x <- 4
	.loc	1 27 16 is_stmt 0       # fibonacci.cc:27:16
	movdqa	%xmm0, (%rsp)
	movdqu	%xmm0, 28(%rsp)
	movdqa	%xmm0, 16(%rsp)
.Ltmp22:
	#DEBUG_VALUE: fib:i <- undef
	#DEBUG_VALUE: fib:i <- [DW_OP_plus_uconst 1, DW_OP_stack_value] undef
	.loc	1 29 14 is_stmt 1       # fibonacci.cc:29:14
	movq	%r14, (%rsp)
	.loc	1 31 14                 # fibonacci.cc:31:14
	movl	$1, 8(%rsp)
	.loc	1 33 10                 # fibonacci.cc:33:10
	movl	$4, %edi
	movq	%rbx, %rsi
	callq	_Z8real_fibiPi
.Ltmp23:
	#DEBUG_VALUE: main:result <- $eax
	.loc	1 43 3                  # fibonacci.cc:43:3
	movl	$.L.str.1, %edi
	movl	%eax, %esi
	xorl	%eax, %eax
.Ltmp24:
	callq	printf
.Ltmp25:
	.loc	1 27 16                 # fibonacci.cc:27:16
	pcmpeqd	%xmm0, %xmm0
.Ltmp26:
	#DEBUG_VALUE: fib:x <- 5
	.loc	1 27 16 is_stmt 0       # fibonacci.cc:27:16
	movdqa	%xmm0, (%rsp)
	movdqu	%xmm0, 28(%rsp)
	movdqa	%xmm0, 16(%rsp)
.Ltmp27:
	#DEBUG_VALUE: fib:i <- undef
	#DEBUG_VALUE: fib:i <- [DW_OP_plus_uconst 1, DW_OP_stack_value] undef
	.loc	1 29 14 is_stmt 1       # fibonacci.cc:29:14
	movq	%r14, (%rsp)
	.loc	1 31 14                 # fibonacci.cc:31:14
	movl	$1, 8(%rsp)
	.loc	1 33 10                 # fibonacci.cc:33:10
	movl	$5, %edi
	movq	%rbx, %rsi
	callq	_Z8real_fibiPi
.Ltmp28:
	#DEBUG_VALUE: main:result <- $eax
	.loc	1 45 3                  # fibonacci.cc:45:3
	movl	$.L.str.2, %edi
	movl	%eax, %esi
	xorl	%eax, %eax
.Ltmp29:
	callq	printf
.Ltmp30:
	.loc	1 27 16                 # fibonacci.cc:27:16
	pcmpeqd	%xmm0, %xmm0
.Ltmp31:
	#DEBUG_VALUE: fib:x <- 6
	.loc	1 27 16 is_stmt 0       # fibonacci.cc:27:16
	movdqa	%xmm0, (%rsp)
	movdqu	%xmm0, 28(%rsp)
	movdqa	%xmm0, 16(%rsp)
.Ltmp32:
	#DEBUG_VALUE: fib:i <- undef
	#DEBUG_VALUE: fib:i <- [DW_OP_plus_uconst 1, DW_OP_stack_value] undef
	.loc	1 29 14 is_stmt 1       # fibonacci.cc:29:14
	movq	%r14, (%rsp)
	.loc	1 31 14                 # fibonacci.cc:31:14
	movl	$1, 8(%rsp)
	.loc	1 33 10                 # fibonacci.cc:33:10
	movl	$6, %edi
	movq	%rbx, %rsi
	callq	_Z8real_fibiPi
.Ltmp33:
	#DEBUG_VALUE: main:result <- $eax
	.loc	1 47 3                  # fibonacci.cc:47:3
	movl	$.L.str.3, %edi
	movl	%eax, %esi
	xorl	%eax, %eax
.Ltmp34:
	callq	printf
.Ltmp35:
	.loc	1 27 16                 # fibonacci.cc:27:16
	pcmpeqd	%xmm0, %xmm0
.Ltmp36:
	#DEBUG_VALUE: fib:x <- 7
	.loc	1 27 16 is_stmt 0       # fibonacci.cc:27:16
	movdqa	%xmm0, (%rsp)
	movdqu	%xmm0, 28(%rsp)
	movdqa	%xmm0, 16(%rsp)
.Ltmp37:
	#DEBUG_VALUE: fib:i <- undef
	#DEBUG_VALUE: fib:i <- [DW_OP_plus_uconst 1, DW_OP_stack_value] undef
	.loc	1 29 14 is_stmt 1       # fibonacci.cc:29:14
	movq	%r14, (%rsp)
	.loc	1 31 14                 # fibonacci.cc:31:14
	movl	$1, 8(%rsp)
	.loc	1 33 10                 # fibonacci.cc:33:10
	movl	$7, %edi
	movq	%rbx, %rsi
	callq	_Z8real_fibiPi
.Ltmp38:
	#DEBUG_VALUE: main:result <- $eax
	.loc	1 49 3                  # fibonacci.cc:49:3
	movl	$.L.str.4, %edi
	movl	%eax, %esi
	xorl	%eax, %eax
.Ltmp39:
	callq	printf
.Ltmp40:
	.loc	1 27 16                 # fibonacci.cc:27:16
	pcmpeqd	%xmm0, %xmm0
.Ltmp41:
	#DEBUG_VALUE: fib:x <- 8
	.loc	1 27 16 is_stmt 0       # fibonacci.cc:27:16
	movdqa	%xmm0, (%rsp)
	movdqu	%xmm0, 28(%rsp)
	movdqa	%xmm0, 16(%rsp)
.Ltmp42:
	#DEBUG_VALUE: fib:i <- undef
	#DEBUG_VALUE: fib:i <- [DW_OP_plus_uconst 1, DW_OP_stack_value] undef
	.loc	1 29 14 is_stmt 1       # fibonacci.cc:29:14
	movq	%r14, (%rsp)
	.loc	1 31 14                 # fibonacci.cc:31:14
	movl	$1, 8(%rsp)
	.loc	1 33 10                 # fibonacci.cc:33:10
	movl	$8, %edi
	movq	%rbx, %rsi
	callq	_Z8real_fibiPi
	movl	%eax, %ebp
.Ltmp43:
	#DEBUG_VALUE: main:result <- $ebp
	.loc	1 51 3                  # fibonacci.cc:51:3
	movl	$.L.str.5, %edi
	movl	%eax, %esi
	xorl	%eax, %eax
	callq	printf
	.loc	1 52 3                  # fibonacci.cc:52:3
	movl	$.L.str.6, %edi
	movl	%ebp, %esi
	xorl	%eax, %eax
	callq	printf
.Ltmp44:
	.loc	1 27 16                 # fibonacci.cc:27:16
	pcmpeqd	%xmm0, %xmm0
.Ltmp45:
	#DEBUG_VALUE: fib:x <- 10
	.loc	1 27 16 is_stmt 0       # fibonacci.cc:27:16
	movdqa	%xmm0, (%rsp)
	movdqu	%xmm0, 28(%rsp)
	movdqa	%xmm0, 16(%rsp)
.Ltmp46:
	#DEBUG_VALUE: fib:i <- undef
	#DEBUG_VALUE: fib:i <- [DW_OP_plus_uconst 1, DW_OP_stack_value] undef
	.loc	1 29 14 is_stmt 1       # fibonacci.cc:29:14
	movq	%r14, (%rsp)
	.loc	1 31 14                 # fibonacci.cc:31:14
	movl	$1, 8(%rsp)
	.loc	1 33 10                 # fibonacci.cc:33:10
	movl	$10, %edi
	movq	%rbx, %rsi
	callq	_Z8real_fibiPi
.Ltmp47:
	#DEBUG_VALUE: main:result <- $eax
	.loc	1 54 3                  # fibonacci.cc:54:3
	movl	$.L.str.7, %edi
	movl	%eax, %esi
	xorl	%eax, %eax
.Ltmp48:
	callq	printf
	.loc	1 56 3                  # fibonacci.cc:56:3
	xorl	%eax, %eax
	addq	$48, %rsp
	.cfi_def_cfa_offset 32
	popq	%rbx
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%rbp
	.cfi_def_cfa_offset 8
	retq
.Ltmp49:
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
.Lskel_string0:
	.asciz	"/usr/local/google3/cmtice/llvm.tot2/build/test/tools/llvm-dwarfdump/X86/Output/statistics-dwo.test.tmp" # string offset=0
.Lskel_string1:
	.asciz	"fib"                   # string offset=71
.Lskel_string2:
	.asciz	"main"                  # string offset=75
.Lskel_string3:
	.asciz	"statistics-fib.split-dwarf.dwo" # string offset=80
	.section	.debug_loc.dwo,"e",@progbits
.Ldebug_loc0:
	.byte	3
	.byte	0
	.long	.Ltmp4-.Lfunc_begin0
	.short	1                       # Loc expr size
	.byte	85                      # super-register DW_OP_reg5
	.byte	0
.Ldebug_loc1:
	.byte	3
	.byte	0
	.long	.Ltmp3-.Lfunc_begin0
	.short	1                       # Loc expr size
	.byte	84                      # DW_OP_reg4
	.byte	3
	.byte	9
	.long	.Ltmp6-.Ltmp3
	.short	1                       # Loc expr size
	.byte	83                      # DW_OP_reg3
	.byte	0
.Ldebug_loc2:
	.byte	3
	.byte	10
	.long	.Lfunc_end0-.Ltmp5
	.short	1                       # Loc expr size
	.byte	80                      # super-register DW_OP_reg0
	.byte	0
.Ldebug_loc3:
	.byte	3
	.byte	1
	.long	.Ltmp13-.Lfunc_begin1
	.short	1                       # Loc expr size
	.byte	85                      # super-register DW_OP_reg5
	.byte	0
.Ldebug_loc4:
	.byte	3
	.byte	2
	.long	.Ltmp16-.Lfunc_begin2
	.short	1                       # Loc expr size
	.byte	85                      # super-register DW_OP_reg5
	.byte	0
.Ldebug_loc5:
	.byte	3
	.byte	2
	.long	.Ltmp17-.Lfunc_begin2
	.short	1                       # Loc expr size
	.byte	84                      # DW_OP_reg4
	.byte	0
.Ldebug_loc6:
	.byte	3
	.byte	11
	.long	.Ltmp19-.Ltmp18
	.short	1                       # Loc expr size
	.byte	80                      # super-register DW_OP_reg0
	.byte	3
	.byte	12
	.long	.Ltmp24-.Ltmp23
	.short	1                       # Loc expr size
	.byte	80                      # super-register DW_OP_reg0
	.byte	3
	.byte	13
	.long	.Ltmp29-.Ltmp28
	.short	1                       # Loc expr size
	.byte	80                      # super-register DW_OP_reg0
	.byte	3
	.byte	14
	.long	.Ltmp34-.Ltmp33
	.short	1                       # Loc expr size
	.byte	80                      # super-register DW_OP_reg0
	.byte	3
	.byte	15
	.long	.Ltmp39-.Ltmp38
	.short	1                       # Loc expr size
	.byte	80                      # super-register DW_OP_reg0
	.byte	3
	.byte	16
	.long	.Ltmp47-.Ltmp43
	.short	1                       # Loc expr size
	.byte	86                      # super-register DW_OP_reg6
	.byte	3
	.byte	17
	.long	.Ltmp48-.Ltmp47
	.short	1                       # Loc expr size
	.byte	80                      # super-register DW_OP_reg0
	.byte	0
.Ldebug_loc7:
	.byte	3
	.byte	3
	.long	.Lfunc_end2-.Ltmp21
	.short	3                       # Loc expr size
	.byte	17                      # DW_OP_consts
	.byte	4                       # 4
	.byte	159                     # DW_OP_stack_value
	.byte	0
.Ldebug_loc8:
	.byte	3
	.byte	4
	.long	.Lfunc_end2-.Ltmp26
	.short	3                       # Loc expr size
	.byte	17                      # DW_OP_consts
	.byte	5                       # 5
	.byte	159                     # DW_OP_stack_value
	.byte	0
.Ldebug_loc9:
	.byte	3
	.byte	5
	.long	.Lfunc_end2-.Ltmp31
	.short	3                       # Loc expr size
	.byte	17                      # DW_OP_consts
	.byte	6                       # 6
	.byte	159                     # DW_OP_stack_value
	.byte	0
.Ldebug_loc10:
	.byte	3
	.byte	6
	.long	.Lfunc_end2-.Ltmp36
	.short	3                       # Loc expr size
	.byte	17                      # DW_OP_consts
	.byte	7                       # 7
	.byte	159                     # DW_OP_stack_value
	.byte	0
.Ldebug_loc11:
	.byte	3
	.byte	7
	.long	.Lfunc_end2-.Ltmp41
	.short	3                       # Loc expr size
	.byte	17                      # DW_OP_consts
	.byte	8                       # 8
	.byte	159                     # DW_OP_stack_value
	.byte	0
.Ldebug_loc12:
	.byte	3
	.byte	8
	.long	.Lfunc_end2-.Ltmp45
	.short	3                       # Loc expr size
	.byte	17                      # DW_OP_consts
	.byte	10                      # 10
	.byte	159                     # DW_OP_stack_value
	.byte	0
	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	16                      # DW_AT_stmt_list
	.byte	23                      # DW_FORM_sec_offset
#	.byte	27                      # DW_AT_comp_dir
#	.byte	14                      # DW_FORM_strp
	.ascii	"\264B"                 # DW_AT_GNU_pubnames
	.byte	25                      # DW_FORM_flag_present
	.ascii	"\260B"                 # DW_AT_GNU_dwo_name
	.byte	14                      # DW_FORM_strp
	.ascii	"\261B"                 # DW_AT_GNU_dwo_id
	.byte	7                       # DW_FORM_data8
	.ascii	"\262B"                 # DW_AT_GNU_ranges_base
	.byte	23                      # DW_FORM_sec_offset
	.ascii	"\263B"                 # DW_AT_GNU_addr_base
	.byte	23                      # DW_FORM_sec_offset
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	2                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	3                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	1                       # DW_CHILDREN_yes
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	4                       # Abbreviation Code
	.byte	29                      # DW_TAG_inlined_subroutine
	.byte	0                       # DW_CHILDREN_no
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
	.byte	5                       # Abbreviation Code
	.byte	29                      # DW_TAG_inlined_subroutine
	.byte	0                       # DW_CHILDREN_no
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
	.byte	0                       # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                       # DWARF version number
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] 0xb:0xbe DW_TAG_compile_unit
	.long	.Lline_table_start0     # DW_AT_stmt_list
#	.long	.Lskel_string0          # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.long	.Lskel_string3          # DW_AT_GNU_dwo_name
	.quad	-7268627715780183436    # DW_AT_GNU_dwo_id
	.long	.debug_ranges           # DW_AT_GNU_ranges_base
	.long	.Laddr_table_base0      # DW_AT_GNU_addr_base
	.quad	.Lfunc_begin0           # DW_AT_low_pc
	.long	.Lfunc_end2-.Lfunc_begin0 # DW_AT_high_pc
	.byte	2                       # Abbrev [2] 0x34:0x5 DW_TAG_subprogram
	.long	.Lskel_string1          # DW_AT_name
	.byte	3                       # Abbrev [3] 0x39:0x8f DW_TAG_subprogram
	.quad	.Lfunc_begin2           # DW_AT_low_pc
	.long	.Lfunc_end2-.Lfunc_begin2 # DW_AT_high_pc
	.long	.Lskel_string2          # DW_AT_name
	.byte	4                       # Abbrev [4] 0x4a:0xb DW_TAG_inlined_subroutine
	.long	52                      # DW_AT_abstract_origin
	.long	.Ldebug_ranges1         # DW_AT_ranges
	.byte	1                       # DW_AT_call_file
	.byte	40                      # DW_AT_call_line
	.byte	5                       # Abbrev [5] 0x55:0x13 DW_TAG_inlined_subroutine
	.long	52                      # DW_AT_abstract_origin
	.quad	.Ltmp21                 # DW_AT_low_pc
	.long	.Ltmp23-.Ltmp21         # DW_AT_high_pc
	.byte	1                       # DW_AT_call_file
	.byte	42                      # DW_AT_call_line
	.byte	5                       # Abbrev [5] 0x68:0x13 DW_TAG_inlined_subroutine
	.long	52                      # DW_AT_abstract_origin
	.quad	.Ltmp26                 # DW_AT_low_pc
	.long	.Ltmp28-.Ltmp26         # DW_AT_high_pc
	.byte	1                       # DW_AT_call_file
	.byte	44                      # DW_AT_call_line
	.byte	5                       # Abbrev [5] 0x7b:0x13 DW_TAG_inlined_subroutine
	.long	52                      # DW_AT_abstract_origin
	.quad	.Ltmp31                 # DW_AT_low_pc
	.long	.Ltmp33-.Ltmp31         # DW_AT_high_pc
	.byte	1                       # DW_AT_call_file
	.byte	46                      # DW_AT_call_line
	.byte	5                       # Abbrev [5] 0x8e:0x13 DW_TAG_inlined_subroutine
	.long	52                      # DW_AT_abstract_origin
	.quad	.Ltmp36                 # DW_AT_low_pc
	.long	.Ltmp38-.Ltmp36         # DW_AT_high_pc
	.byte	1                       # DW_AT_call_file
	.byte	48                      # DW_AT_call_line
	.byte	5                       # Abbrev [5] 0xa1:0x13 DW_TAG_inlined_subroutine
	.long	52                      # DW_AT_abstract_origin
	.quad	.Ltmp41                 # DW_AT_low_pc
	.long	.Ltmp43-.Ltmp41         # DW_AT_high_pc
	.byte	1                       # DW_AT_call_file
	.byte	50                      # DW_AT_call_line
	.byte	5                       # Abbrev [5] 0xb4:0x13 DW_TAG_inlined_subroutine
	.long	52                      # DW_AT_abstract_origin
	.quad	.Ltmp45                 # DW_AT_low_pc
	.long	.Ltmp47-.Ltmp45         # DW_AT_high_pc
	.byte	1                       # DW_AT_call_file
	.byte	53                      # DW_AT_call_line
	.byte	0                       # End Of Children Mark
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
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	0
	.quad	0
.Ldebug_ranges1:
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
	.quad	.Ltmp44-.Lfunc_begin0
	.quad	.Ltmp45-.Lfunc_begin0
	.quad	0
	.quad	0
	.section	.debug_macinfo,"",@progbits
	.byte	0                       # End Of Macro List Mark
	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"_Z3fibi"               # string offset=0
.Linfo_string1:
	.asciz	"fib"                   # string offset=8
.Linfo_string2:
	.asciz	"int"                   # string offset=12
.Linfo_string3:
	.asciz	"x"                     # string offset=16
.Linfo_string4:
	.asciz	"answers"               # string offset=18
.Linfo_string5:
	.asciz	"__ARRAY_SIZE_TYPE__"   # string offset=26
.Linfo_string6:
	.asciz	"i"                     # string offset=46
.Linfo_string7:
	.asciz	"_Z8real_fibiPi"        # string offset=48
.Linfo_string8:
	.asciz	"real_fib"              # string offset=63
.Linfo_string9:
	.asciz	"main"                  # string offset=72
.Linfo_string10:
	.asciz	"result"                # string offset=77
.Linfo_string11:
	.asciz	"argc"                  # string offset=84
.Linfo_string12:
	.asciz	"argv"                  # string offset=89
.Linfo_string13:
	.asciz	"char"                  # string offset=94
.Linfo_string14:
	.asciz	"clang version 9.0.0 (trunk 358316)" # string offset=99
.Linfo_string15:
	.asciz	"fibonacci.cc"          # string offset=134
.Linfo_string16:
	.asciz	"statistics-fib.split-dwarf.dwo" # string offset=147
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	0
	.long	8
	.long	12
	.long	16
	.long	18
	.long	26
	.long	46
	.long	48
	.long	63
	.long	72
	.long	77
	.long	84
	.long	89
	.long	94
	.long	99
	.long	134
	.long	147
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	4                       # DWARF version number
	.long	0                       # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] 0xb:0x1d6 DW_TAG_compile_unit
	.byte	14                      # DW_AT_producer
	.short	4                       # DW_AT_language
	.byte	15                      # DW_AT_name
	.byte	16                      # DW_AT_GNU_dwo_name
	.quad	-7268627715780183436    # DW_AT_GNU_dwo_id
	.byte	2                       # Abbrev [2] 0x19:0x35 DW_TAG_subprogram
	.byte	0                       # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
	.byte	1                       # DW_AT_frame_base
	.byte	87
	.byte	7                       # DW_AT_linkage_name
	.byte	8                       # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	4                       # DW_AT_decl_line
	.long	148                     # DW_AT_type
                                        # DW_AT_external
	.byte	3                       # Abbrev [3] 0x29:0xc DW_TAG_formal_parameter
	.long	.Ldebug_loc0-.debug_loc.dwo # DW_AT_location
	.byte	3                       # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	4                       # DW_AT_decl_line
	.long	148                     # DW_AT_type
	.byte	3                       # Abbrev [3] 0x35:0xc DW_TAG_formal_parameter
	.long	.Ldebug_loc1-.debug_loc.dwo # DW_AT_location
	.byte	4                       # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	4                       # DW_AT_decl_line
	.long	461                     # DW_AT_type
	.byte	4                       # Abbrev [4] 0x41:0xc DW_TAG_variable
	.long	.Ldebug_loc2-.debug_loc.dwo # DW_AT_location
	.byte	10                      # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	6                       # DW_AT_decl_line
	.long	148                     # DW_AT_type
	.byte	0                       # End Of Children Mark
	.byte	5                       # Abbrev [5] 0x4e:0x23 DW_TAG_subprogram
	.byte	1                       # DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1 # DW_AT_high_pc
	.byte	1                       # DW_AT_frame_base
	.byte	87
	.long	113                     # DW_AT_abstract_origin
	.byte	6                       # Abbrev [6] 0x5a:0x9 DW_TAG_formal_parameter
	.long	.Ldebug_loc3-.debug_loc.dwo # DW_AT_location
	.long	123                     # DW_AT_abstract_origin
	.byte	7                       # Abbrev [7] 0x63:0x8 DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	0
	.long	131                     # DW_AT_abstract_origin
	.byte	8                       # Abbrev [8] 0x6b:0x5 DW_TAG_variable
	.long	139                     # DW_AT_abstract_origin
	.byte	0                       # End Of Children Mark
	.byte	9                       # Abbrev [9] 0x71:0x23 DW_TAG_subprogram
	.byte	0                       # DW_AT_linkage_name
	.byte	1                       # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	18                      # DW_AT_decl_line
	.long	148                     # DW_AT_type
                                        # DW_AT_external
	.byte	1                       # DW_AT_inline
	.byte	10                      # Abbrev [10] 0x7b:0x8 DW_TAG_formal_parameter
	.byte	3                       # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	18                      # DW_AT_decl_line
	.long	148                     # DW_AT_type
	.byte	11                      # Abbrev [11] 0x83:0x8 DW_TAG_variable
	.byte	4                       # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	20                      # DW_AT_decl_line
	.long	152                     # DW_AT_type
	.byte	11                      # Abbrev [11] 0x8b:0x8 DW_TAG_variable
	.byte	6                       # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	21                      # DW_AT_decl_line
	.long	148                     # DW_AT_type
	.byte	0                       # End Of Children Mark
	.byte	12                      # Abbrev [12] 0x94:0x4 DW_TAG_base_type
	.byte	2                       # DW_AT_name
	.byte	5                       # DW_AT_encoding
	.byte	4                       # DW_AT_byte_size
	.byte	13                      # Abbrev [13] 0x98:0xc DW_TAG_array_type
	.long	148                     # DW_AT_type
	.byte	14                      # Abbrev [14] 0x9d:0x6 DW_TAG_subrange_type
	.long	164                     # DW_AT_type
	.byte	11                      # DW_AT_count
	.byte	0                       # End Of Children Mark
	.byte	15                      # Abbrev [15] 0xa4:0x4 DW_TAG_base_type
	.byte	5                       # DW_AT_name
	.byte	8                       # DW_AT_byte_size
	.byte	7                       # DW_AT_encoding
	.byte	16                      # Abbrev [16] 0xa8:0x125 DW_TAG_subprogram
	.byte	2                       # DW_AT_low_pc
	.long	.Lfunc_end2-.Lfunc_begin2 # DW_AT_high_pc
	.byte	1                       # DW_AT_frame_base
	.byte	87
	.byte	9                       # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	36                      # DW_AT_decl_line
	.long	148                     # DW_AT_type
                                        # DW_AT_external
	.byte	3                       # Abbrev [3] 0xb7:0xc DW_TAG_formal_parameter
	.long	.Ldebug_loc4-.debug_loc.dwo # DW_AT_location
	.byte	11                      # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	36                      # DW_AT_decl_line
	.long	148                     # DW_AT_type
	.byte	3                       # Abbrev [3] 0xc3:0xc DW_TAG_formal_parameter
	.long	.Ldebug_loc5-.debug_loc.dwo # DW_AT_location
	.byte	12                      # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	36                      # DW_AT_decl_line
	.long	466                     # DW_AT_type
	.byte	4                       # Abbrev [4] 0xcf:0xc DW_TAG_variable
	.long	.Ldebug_loc6-.debug_loc.dwo # DW_AT_location
	.byte	10                      # DW_AT_name
	.byte	1                       # DW_AT_decl_file
	.byte	38                      # DW_AT_decl_line
	.long	148                     # DW_AT_type
	.byte	17                      # Abbrev [17] 0xdb:0x1f DW_TAG_inlined_subroutine
	.long	113                     # DW_AT_abstract_origin
	.long	.Ldebug_ranges0-.debug_ranges # DW_AT_ranges
	.byte	1                       # DW_AT_call_file
	.byte	40                      # DW_AT_call_line
	.byte	18                      # Abbrev [18] 0xe6:0x6 DW_TAG_formal_parameter
	.byte	3                       # DW_AT_const_value
	.long	123                     # DW_AT_abstract_origin
	.byte	7                       # Abbrev [7] 0xec:0x8 DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	0
	.long	131                     # DW_AT_abstract_origin
	.byte	8                       # Abbrev [8] 0xf4:0x5 DW_TAG_variable
	.long	139                     # DW_AT_abstract_origin
	.byte	0                       # End Of Children Mark
	.byte	19                      # Abbrev [19] 0xfa:0x23 DW_TAG_inlined_subroutine
	.long	113                     # DW_AT_abstract_origin
	.byte	3                       # DW_AT_low_pc
	.long	.Ltmp23-.Ltmp21         # DW_AT_high_pc
	.byte	1                       # DW_AT_call_file
	.byte	42                      # DW_AT_call_line
	.byte	6                       # Abbrev [6] 0x106:0x9 DW_TAG_formal_parameter
	.long	.Ldebug_loc7-.debug_loc.dwo # DW_AT_location
	.long	123                     # DW_AT_abstract_origin
	.byte	7                       # Abbrev [7] 0x10f:0x8 DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	0
	.long	131                     # DW_AT_abstract_origin
	.byte	8                       # Abbrev [8] 0x117:0x5 DW_TAG_variable
	.long	139                     # DW_AT_abstract_origin
	.byte	0                       # End Of Children Mark
	.byte	19                      # Abbrev [19] 0x11d:0x23 DW_TAG_inlined_subroutine
	.long	113                     # DW_AT_abstract_origin
	.byte	4                       # DW_AT_low_pc
	.long	.Ltmp28-.Ltmp26         # DW_AT_high_pc
	.byte	1                       # DW_AT_call_file
	.byte	44                      # DW_AT_call_line
	.byte	6                       # Abbrev [6] 0x129:0x9 DW_TAG_formal_parameter
	.long	.Ldebug_loc8-.debug_loc.dwo # DW_AT_location
	.long	123                     # DW_AT_abstract_origin
	.byte	7                       # Abbrev [7] 0x132:0x8 DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	0
	.long	131                     # DW_AT_abstract_origin
	.byte	8                       # Abbrev [8] 0x13a:0x5 DW_TAG_variable
	.long	139                     # DW_AT_abstract_origin
	.byte	0                       # End Of Children Mark
	.byte	19                      # Abbrev [19] 0x140:0x23 DW_TAG_inlined_subroutine
	.long	113                     # DW_AT_abstract_origin
	.byte	5                       # DW_AT_low_pc
	.long	.Ltmp33-.Ltmp31         # DW_AT_high_pc
	.byte	1                       # DW_AT_call_file
	.byte	46                      # DW_AT_call_line
	.byte	6                       # Abbrev [6] 0x14c:0x9 DW_TAG_formal_parameter
	.long	.Ldebug_loc9-.debug_loc.dwo # DW_AT_location
	.long	123                     # DW_AT_abstract_origin
	.byte	7                       # Abbrev [7] 0x155:0x8 DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	0
	.long	131                     # DW_AT_abstract_origin
	.byte	8                       # Abbrev [8] 0x15d:0x5 DW_TAG_variable
	.long	139                     # DW_AT_abstract_origin
	.byte	0                       # End Of Children Mark
	.byte	19                      # Abbrev [19] 0x163:0x23 DW_TAG_inlined_subroutine
	.long	113                     # DW_AT_abstract_origin
	.byte	6                       # DW_AT_low_pc
	.long	.Ltmp38-.Ltmp36         # DW_AT_high_pc
	.byte	1                       # DW_AT_call_file
	.byte	48                      # DW_AT_call_line
	.byte	6                       # Abbrev [6] 0x16f:0x9 DW_TAG_formal_parameter
	.long	.Ldebug_loc10-.debug_loc.dwo # DW_AT_location
	.long	123                     # DW_AT_abstract_origin
	.byte	7                       # Abbrev [7] 0x178:0x8 DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	0
	.long	131                     # DW_AT_abstract_origin
	.byte	8                       # Abbrev [8] 0x180:0x5 DW_TAG_variable
	.long	139                     # DW_AT_abstract_origin
	.byte	0                       # End Of Children Mark
	.byte	19                      # Abbrev [19] 0x186:0x23 DW_TAG_inlined_subroutine
	.long	113                     # DW_AT_abstract_origin
	.byte	7                       # DW_AT_low_pc
	.long	.Ltmp43-.Ltmp41         # DW_AT_high_pc
	.byte	1                       # DW_AT_call_file
	.byte	50                      # DW_AT_call_line
	.byte	6                       # Abbrev [6] 0x192:0x9 DW_TAG_formal_parameter
	.long	.Ldebug_loc11-.debug_loc.dwo # DW_AT_location
	.long	123                     # DW_AT_abstract_origin
	.byte	7                       # Abbrev [7] 0x19b:0x8 DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	0
	.long	131                     # DW_AT_abstract_origin
	.byte	8                       # Abbrev [8] 0x1a3:0x5 DW_TAG_variable
	.long	139                     # DW_AT_abstract_origin
	.byte	0                       # End Of Children Mark
	.byte	19                      # Abbrev [19] 0x1a9:0x23 DW_TAG_inlined_subroutine
	.long	113                     # DW_AT_abstract_origin
	.byte	8                       # DW_AT_low_pc
	.long	.Ltmp47-.Ltmp45         # DW_AT_high_pc
	.byte	1                       # DW_AT_call_file
	.byte	53                      # DW_AT_call_line
	.byte	6                       # Abbrev [6] 0x1b5:0x9 DW_TAG_formal_parameter
	.long	.Ldebug_loc12-.debug_loc.dwo # DW_AT_location
	.long	123                     # DW_AT_abstract_origin
	.byte	7                       # Abbrev [7] 0x1be:0x8 DW_TAG_variable
	.byte	2                       # DW_AT_location
	.byte	145
	.byte	0
	.long	131                     # DW_AT_abstract_origin
	.byte	8                       # Abbrev [8] 0x1c6:0x5 DW_TAG_variable
	.long	139                     # DW_AT_abstract_origin
	.byte	0                       # End Of Children Mark
	.byte	0                       # End Of Children Mark
	.byte	20                      # Abbrev [20] 0x1cd:0x5 DW_TAG_pointer_type
	.long	148                     # DW_AT_type
	.byte	20                      # Abbrev [20] 0x1d2:0x5 DW_TAG_pointer_type
	.long	471                     # DW_AT_type
	.byte	20                      # Abbrev [20] 0x1d7:0x5 DW_TAG_pointer_type
	.long	476                     # DW_AT_type
	.byte	12                      # Abbrev [12] 0x1dc:0x4 DW_TAG_base_type
	.byte	13                      # DW_AT_name
	.byte	6                       # DW_AT_encoding
	.byte	1                       # DW_AT_byte_size
	.byte	0                       # End Of Children Mark
.Ldebug_info_dwo_end0:
	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	37                      # DW_AT_producer
	.ascii	"\202>"                 # DW_FORM_GNU_str_index
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	3                       # DW_AT_name
	.ascii	"\202>"                 # DW_FORM_GNU_str_index
	.ascii	"\260B"                 # DW_AT_GNU_dwo_name
	.ascii	"\202>"                 # DW_FORM_GNU_str_index
	.ascii	"\261B"                 # DW_AT_GNU_dwo_id
	.byte	7                       # DW_FORM_data8
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	2                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	1                       # DW_CHILDREN_yes
	.byte	17                      # DW_AT_low_pc
	.ascii	"\201>"                 # DW_FORM_GNU_addr_index
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	64                      # DW_AT_frame_base
	.byte	24                      # DW_FORM_exprloc
	.byte	110                     # DW_AT_linkage_name
	.ascii	"\202>"                 # DW_FORM_GNU_str_index
	.byte	3                       # DW_AT_name
	.ascii	"\202>"                 # DW_FORM_GNU_str_index
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
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
	.ascii	"\202>"                 # DW_FORM_GNU_str_index
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
	.ascii	"\202>"                 # DW_FORM_GNU_str_index
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
	.ascii	"\201>"                 # DW_FORM_GNU_addr_index
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
	.byte	110                     # DW_AT_linkage_name
	.ascii	"\202>"                 # DW_FORM_GNU_str_index
	.byte	3                       # DW_AT_name
	.ascii	"\202>"                 # DW_FORM_GNU_str_index
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
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
	.ascii	"\202>"                 # DW_FORM_GNU_str_index
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
	.ascii	"\202>"                 # DW_FORM_GNU_str_index
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
	.ascii	"\202>"                 # DW_FORM_GNU_str_index
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
	.ascii	"\202>"                 # DW_FORM_GNU_str_index
	.byte	11                      # DW_AT_byte_size
	.byte	11                      # DW_FORM_data1
	.byte	62                      # DW_AT_encoding
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	16                      # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	1                       # DW_CHILDREN_yes
	.byte	17                      # DW_AT_low_pc
	.ascii	"\201>"                 # DW_FORM_GNU_addr_index
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	64                      # DW_AT_frame_base
	.byte	24                      # DW_FORM_exprloc
	.byte	3                       # DW_AT_name
	.ascii	"\202>"                 # DW_FORM_GNU_str_index
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	63                      # DW_AT_external
	.byte	25                      # DW_FORM_flag_present
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	17                      # Abbreviation Code
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
	.byte	18                      # Abbreviation Code
	.byte	5                       # DW_TAG_formal_parameter
	.byte	0                       # DW_CHILDREN_no
	.byte	28                      # DW_AT_const_value
	.byte	13                      # DW_FORM_sdata
	.byte	49                      # DW_AT_abstract_origin
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	19                      # Abbreviation Code
	.byte	29                      # DW_TAG_inlined_subroutine
	.byte	1                       # DW_CHILDREN_yes
	.byte	49                      # DW_AT_abstract_origin
	.byte	19                      # DW_FORM_ref4
	.byte	17                      # DW_AT_low_pc
	.ascii	"\201>"                 # DW_FORM_GNU_addr_index
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	88                      # DW_AT_call_file
	.byte	11                      # DW_FORM_data1
	.byte	89                      # DW_AT_call_line
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	20                      # Abbreviation Code
	.byte	15                      # DW_TAG_pointer_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
	.section	.debug_addr,"",@progbits
.Laddr_table_base0:
	.quad	.Lfunc_begin0
	.quad	.Lfunc_begin1
	.quad	.Lfunc_begin2
	.quad	.Ltmp21
	.quad	.Ltmp26
	.quad	.Ltmp31
	.quad	.Ltmp36
	.quad	.Ltmp41
	.quad	.Ltmp45
	.quad	.Ltmp3
	.quad	.Ltmp5
	.quad	.Ltmp18
	.quad	.Ltmp23
	.quad	.Ltmp28
	.quad	.Ltmp33
	.quad	.Ltmp38
	.quad	.Ltmp43
	.quad	.Ltmp47
	.section	.debug_gnu_pubnames,"",@progbits
	.long	.LpubNames_end0-.LpubNames_begin0 # Length of Public Names Info
.LpubNames_begin0:
	.short	2                       # DWARF Version
	.long	.Lcu_begin0             # Offset of Compilation Unit Info
	.long	201                     # Compilation Unit Length
	.long	113                     # DIE offset
	.byte	48                      # Attributes: FUNCTION, EXTERNAL
	.asciz	"fib"                   # External Name
	.long	25                      # DIE offset
	.byte	48                      # Attributes: FUNCTION, EXTERNAL
	.asciz	"real_fib"              # External Name
	.long	168                     # DIE offset
	.byte	48                      # Attributes: FUNCTION, EXTERNAL
	.asciz	"main"                  # External Name
	.long	0                       # End Mark
.LpubNames_end0:
	.section	.debug_gnu_pubtypes,"",@progbits
	.long	.LpubTypes_end0-.LpubTypes_begin0 # Length of Public Types Info
.LpubTypes_begin0:
	.short	2                       # DWARF Version
	.long	.Lcu_begin0             # Offset of Compilation Unit Info
	.long	201                     # Compilation Unit Length
	.long	148                     # DIE offset
	.byte	144                     # Attributes: TYPE, STATIC
	.asciz	"int"                   # External Name
	.long	476                     # DIE offset
	.byte	144                     # Attributes: TYPE, STATIC
	.asciz	"char"                  # External Name
	.long	0                       # End Mark
.LpubTypes_end0:

	.ident	"clang version 9.0.0 (trunk 358316)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
