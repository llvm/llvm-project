# RUN: llvm-mc %s -filetype obj -triple x86_64-apple-darwin -o - \
# RUN: | not llvm-dwarfdump -verify - \
# RUN: | FileCheck %s

# CHECK: Verifying .debug_info Unit Header Chain...
# CHECK: warning: DW_TAG_compile_unit has DW_CHILDREN_yes but DIE has no children:
# CHECK: error: Invalid address range [0x0000000000000007, 0x0000000000000006)

	.section	__TEXT,__text,regular,pure_instructions
	.macosx_version_min 10, 12
	.globl	_foo                    ## -- Begin function foo
	.p2align	4, 0x90
_foo:                                   ## @foo
Lfunc_begin0:
	.file	1 "basic.c"
	.loc	1 1 0                   ## basic.c:1:0
	.cfi_startproc
## %bb.0:                               ## %entry
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
Ltmp0:
	.loc	1 1 17 prologue_end     ## basic.c:1:17
	popq	%rbp
	retq
Ltmp1:
Lfunc_end0:
	.cfi_endproc
                                        ## -- End function
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.asciz	"clang version 6.0.0 (trunk 308773) (llvm/trunk 308774)" ## string offset=0
	.asciz	"basic.c"               ## string offset=55
	.asciz	"/Users/sgravani/Development/tests" ## string offset=63
	.asciz	"foo"                   ## string offset=97
	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                       ## Abbreviation Code
	.byte	17                      ## DW_TAG_compile_unit
	.byte	1                       ## DW_CHILDREN_yes
	.byte	37                      ## DW_AT_producer
	.byte	14                      ## DW_FORM_strp
	.byte	19                      ## DW_AT_language
	.byte	5                       ## DW_FORM_data2
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	16                      ## DW_AT_stmt_list
	.byte	6                       ## DW_FORM_data4
	.byte	27                      ## DW_AT_comp_dir
	.byte	14                      ## DW_FORM_strp
	.byte	17                      ## DW_AT_low_pc
	.byte	1                       ## DW_FORM_addr
	.byte	18                      ## DW_AT_high_pc
	.byte	1                       ## DW_FORM_addr
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
	.long	43                      ## Length of Unit
	.short	3                       ## DWARF version number
Lset0 = Lsection_abbrev-Lsection_abbrev ## Offset Into Abbrev. Section
	.long	Lset0
	.byte	8                       ## Address Size (in bytes)
	.byte	1                       ## Abbrev [1] 0xb:0x3f DW_TAG_compile_unit
	.long	0                       ## DW_AT_producer
	.short	12                      ## DW_AT_language
	.long	55                      ## DW_AT_name
Lset1 = Lline_table_start0-Lsection_line ## DW_AT_stmt_list
	.long	Lset1
	.long	63                      ## DW_AT_comp_dir
	.quad	7        				## DW_AT_low_pc -- error: Invalid address range [0x0000000000000007, 0x0000000000000006)
	.quad	6             			## DW_AT_high_pc
	.byte	0                       ## End Of Children Mark
.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
