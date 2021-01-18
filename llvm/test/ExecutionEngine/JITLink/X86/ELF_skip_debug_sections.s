# REQUIRES: asserts
# RUN: llvm-mc -triple=x86_64-pc-linux-gnu -filetype=obj -o %t %s
# RUN: llvm-jitlink -debug-only=jitlink -noexec %t 2>&1 | FileCheck %s
#
# Check that debug sections are not emitted.
#
# CHECK: .debug_info is a debug section: No graph section will be created.

	.text
	.file	"ELF_skip_debug_sections.c"
	.globl	foo
	.p2align	4, 0x90
	.type	foo,@function
foo:
.Lfunc_begin0:
	.file	1 "/tmp" "ELF_skip_debug_sections.c"
	.loc	1 1 0
	.cfi_startproc

	.loc	1 2 3 prologue_end
	movl	$42, %eax
	retq
.Ltmp0:
.Lfunc_end0:
	.size	foo, .Lfunc_end0-foo
	.cfi_endproc

	.globl	main
	.p2align	4, 0x90
	.type	main,@function
main:
.Lfunc_begin1:
	.loc	1 5 0
	.cfi_startproc



	.loc	1 6 3 prologue_end
	movl	$42, %eax
	retq
.Ltmp1:
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc

	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 10.0.0-4ubuntu1 "
.Linfo_string1:
	.asciz	"ELF_skip_debug_sections.c"
.Linfo_string2:
	.asciz	"/tmp"
.Linfo_string3:
	.asciz	"foo"
.Linfo_string4:
	.asciz	"int"
.Linfo_string5:
	.asciz	"main"
.Linfo_string6:
	.asciz	"argc"
.Linfo_string7:
	.asciz	"argv"
.Linfo_string8:
	.asciz	"char"
	.section	.debug_abbrev,"",@progbits
	.byte	1
	.byte	17
	.byte	1
	.byte	37
	.byte	14
	.byte	19
	.byte	5
	.byte	3
	.byte	14
	.byte	16
	.byte	23
	.byte	27
	.byte	14
	.byte	17
	.byte	1
	.byte	18
	.byte	6
	.byte	0
	.byte	0
	.byte	2
	.byte	46
	.byte	0
	.byte	17
	.byte	1
	.byte	18
	.byte	6
	.byte	64
	.byte	24
	.ascii	"\227B"
	.byte	25
	.byte	3
	.byte	14
	.byte	58
	.byte	11
	.byte	59
	.byte	11
	.byte	39
	.byte	25
	.byte	73
	.byte	19
	.byte	63
	.byte	25
	.byte	0
	.byte	0
	.byte	3
	.byte	46
	.byte	1
	.byte	17
	.byte	1
	.byte	18
	.byte	6
	.byte	64
	.byte	24
	.ascii	"\227B"
	.byte	25
	.byte	3
	.byte	14
	.byte	58
	.byte	11
	.byte	59
	.byte	11
	.byte	39
	.byte	25
	.byte	73
	.byte	19
	.byte	63
	.byte	25
	.byte	0
	.byte	0
	.byte	4
	.byte	5
	.byte	0
	.byte	2
	.byte	24
	.byte	3
	.byte	14
	.byte	58
	.byte	11
	.byte	59
	.byte	11
	.byte	73
	.byte	19
	.byte	0
	.byte	0
	.byte	5
	.byte	36
	.byte	0
	.byte	3
	.byte	14
	.byte	62
	.byte	11
	.byte	11
	.byte	11
	.byte	0
	.byte	0
	.byte	6
	.byte	15
	.byte	0
	.byte	73
	.byte	19
	.byte	0
	.byte	0
	.byte	0
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0
.Ldebug_info_start0:
	.short	4
	.long	.debug_abbrev
	.byte	8
	.byte	1
	.long	.Linfo_string0
	.short	12
	.long	.Linfo_string1
	.long	.Lline_table_start0
	.long	.Linfo_string2
	.quad	.Lfunc_begin0
	.long	.Lfunc_end1-.Lfunc_begin0
	.byte	2
	.quad	.Lfunc_begin0
	.long	.Lfunc_end0-.Lfunc_begin0
	.byte	1
	.byte	87

	.long	.Linfo_string3
	.byte	1
	.byte	1

	.long	119

	.byte	3
	.quad	.Lfunc_begin1
	.long	.Lfunc_end1-.Lfunc_begin1
	.byte	1
	.byte	87

	.long	.Linfo_string5
	.byte	1
	.byte	5

	.long	119

	.byte	4
	.byte	1
	.byte	85
	.long	.Linfo_string6
	.byte	1
	.byte	5
	.long	119
	.byte	4
	.byte	1
	.byte	84
	.long	.Linfo_string7
	.byte	1
	.byte	5
	.long	126
	.byte	0
	.byte	5
	.long	.Linfo_string4
	.byte	5
	.byte	4
	.byte	6
	.long	131
	.byte	6
	.long	136
	.byte	5
	.long	.Linfo_string8
	.byte	6
	.byte	1
	.byte	0
.Ldebug_info_end0:
	.ident	"clang version 10.0.0-4ubuntu1 "
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
