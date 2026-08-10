# RUN: llvm-mc -dwarf-version=5 %s -filetype obj -triple x86_64-apple-darwin -o - \
# RUN:   | llvm-dwarfdump - | FileCheck %s

# CHECK: DW_AT_location  (DW_OP_regval_type RSI (0x0000003a) "int")

	.section	__TEXT,__text,regular,pure_instructions
	.globl	_f                      ## -- Begin function f
	.p2align	4, 0x90
_f:                                     ## @f
Lfunc_begin0:
	.file	0 "/Volumes/Data/llvm-project" "/tmp/t.c" md5 0xe111397c9e6224481f168e197b7fac99
	.loc	0 1 0                   ## /tmp/t.c:1:0
	.cfi_startproc
## %bb.0:
	pushq	%rbp
	.cfi_def_cfa_offset 16
	.cfi_offset %rbp, -16
	movq	%rsp, %rbp
	.cfi_def_cfa_register %rbp
Ltmp0:
	.loc	0 2 7 prologue_end      ## /tmp/t.c:2:7
	movl	$23, -4(%rbp)
	.loc	0 3 1                   ## /tmp/t.c:3:1
	popq	%rbp
	retq
Ltmp1:
Lfunc_end0:
	.cfi_endproc
                                        ## -- End function
	.section	__DWARF,__debug_str_offs,regular,debug
Lsection_str_off:
	.long	28
	.short	5
	.short	0
Lstr_offsets_base0:
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.asciz	"clang                                          " ## string offset=0
	.asciz	"/tmp/t.c"              ## string offset=48
	.asciz	"/Volumes/Data/llvm-project" ## string offset=57
	.asciz	"f"                     ## string offset=84
	.asciz	"x"                     ## string offset=86
	.asciz	"int"                   ## string offset=88
	.section	__DWARF,__debug_str_offs,regular,debug
	.long	0
	.long	48
	.long	57
	.long	84
	.long	86
	.long	88
	.section	__DWARF,__debug_abbrev,regular,debug
Lsection_abbrev:
	.byte	1                       ## Abbreviation Code
	.byte	17                      ## DW_TAG_compile_unit
	.byte	1                       ## DW_CHILDREN_yes
	.byte	37                      ## DW_AT_producer
	.byte	37                      ## DW_FORM_strx1
	.byte	19                      ## DW_AT_language
	.byte	5                       ## DW_FORM_data2
	.byte	3                       ## DW_AT_name
	.byte	37                      ## DW_FORM_strx1
	.byte	114                     ## DW_AT_str_offsets_base
	.byte	23                      ## DW_FORM_sec_offset
	.byte	16                      ## DW_AT_stmt_list
	.byte	23                      ## DW_FORM_sec_offset
	.byte	27                      ## DW_AT_comp_dir
	.byte	37                      ## DW_FORM_strx1
	.byte	17                      ## DW_AT_low_pc
	.byte	27                      ## DW_FORM_addrx
	.byte	18                      ## DW_AT_high_pc
	.byte	6                       ## DW_FORM_data4
	.byte	115                     ## DW_AT_addr_base
	.byte	23                      ## DW_FORM_sec_offset
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	2                       ## Abbreviation Code
	.byte	46                      ## DW_TAG_subprogram
	.byte	1                       ## DW_CHILDREN_yes
	.byte	17                      ## DW_AT_low_pc
	.byte	27                      ## DW_FORM_addrx
	.byte	18                      ## DW_AT_high_pc
	.byte	6                       ## DW_FORM_data4
	.byte	64                      ## DW_AT_frame_base
	.byte	24                      ## DW_FORM_exprloc
	.byte	3                       ## DW_AT_name
	.byte	37                      ## DW_FORM_strx1
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	63                      ## DW_AT_external
	.byte	25                      ## DW_FORM_flag_present
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	3                       ## Abbreviation Code
	.byte	52                      ## DW_TAG_variable
	.byte	0                       ## DW_CHILDREN_no
	.byte	2                       ## DW_AT_location
	.byte	24                      ## DW_FORM_exprloc
	.byte	3                       ## DW_AT_name
	.byte	37                      ## DW_FORM_strx1
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	4                       ## Abbreviation Code
	.byte	36                      ## DW_TAG_base_type
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	37                      ## DW_FORM_strx1
	.byte	62                      ## DW_AT_encoding
	.byte	11                      ## DW_FORM_data1
	.byte	11                      ## DW_AT_byte_size
	.byte	11                      ## DW_FORM_data1
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	0                       ## EOM(3)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
.set Lset0, Ldebug_info_end0-Ldebug_info_start0 ## Length of Unit
	.long	Lset0
Ldebug_info_start0:
	.short	5                       ## DWARF version number
	.byte	1                       ## DWARF Unit Type
	.byte	8                       ## Address Size (in bytes)
.set Lset1, Lsection_abbrev-Lsection_abbrev ## Offset Into Abbrev. Section
	.long	Lset1
	.byte	1                       ## Abbrev [1] 0xc:0x33 DW_TAG_compile_unit
	.byte	0                       ## DW_AT_producer
	.short	12                      ## DW_AT_language
	.byte	1                       ## DW_AT_name
.set Lset2, Lstr_offsets_base0-Lsection_str_off ## DW_AT_str_offsets_base
	.long	Lset2
.set Lset3, Lline_table_start0-Lsection_line ## DW_AT_stmt_list
	.long	Lset3
	.byte	2                       ## DW_AT_comp_dir
	.byte	0                       ## DW_AT_low_pc
.set Lset4, Lfunc_end0-Lfunc_begin0     ## DW_AT_high_pc
	.long	Lset4
.set Lset5, Laddr_table_base0-Lsection_info0 ## DW_AT_addr_base
	.long	Lset5
	.byte	2                       ## Abbrev [2] 0x23:0x17 DW_TAG_subprogram
	.byte	0                       ## DW_AT_low_pc
.set Lset6, Lfunc_end0-Lfunc_begin0     ## DW_AT_high_pc
	.long	Lset6
	.byte	1                       ## DW_AT_frame_base
	.byte	86
	.byte	3                       ## DW_AT_name
	.byte	0                       ## DW_AT_decl_file
	.byte	1                       ## DW_AT_decl_line
                                        ## DW_AT_external
	.byte	3                       ## Abbrev [3] 0x2e:0xb DW_TAG_variable
	.byte	3                       ## DW_AT_location
	.byte	0xa5
	.byte	0x04
        .byte   0x3a
	.byte	4                       ## DW_AT_name
	.byte	0                       ## DW_AT_decl_file
	.long	58                      ## DW_AT_type
	.byte	0                       ## End Of Children Mark
	.byte	4                       ## Abbrev [4] 0x3a:0x4 DW_TAG_base_type
	.byte	5                       ## DW_AT_name
	.byte	5                       ## DW_AT_encoding
	.byte	4                       ## DW_AT_byte_size
	.byte	0                       ## End Of Children Mark
Ldebug_info_end0:
	.section	__DWARF,__debug_addr,regular,debug
Lsection_info0:
.set Lset7, Ldebug_addr_end0-Ldebug_addr_start0 ## Length of contribution
	.long	Lset7
Ldebug_addr_start0:
	.short	5                       ## DWARF version number
	.byte	8                       ## Address size
	.byte	0                       ## Segment selector size
Laddr_table_base0:
	.quad	Lfunc_begin0
Ldebug_addr_end0:

.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
