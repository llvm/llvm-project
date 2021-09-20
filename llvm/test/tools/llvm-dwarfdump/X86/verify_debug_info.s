# RUN: llvm-mc %s -filetype obj -triple x86_64-apple-darwin -o - \
# RUN: | not llvm-dwarfdump -v -verify - \
# RUN: | FileCheck %s

# CHECK: error: DIE has invalid DW_AT_stmt_list encoding:{{[[:space:]]}}
# CHECK-NEXT: 0x0000000c: DW_TAG_compile_unit [1] *
# CHECK-NEXT: DW_AT_producer [DW_FORM_strp]	( .debug_str[0x00000000] = "clang version 5.0.0 (trunk 308185) (llvm/trunk 308186)")
# CHECK-NEXT: DW_AT_language [DW_FORM_data2]	(DW_LANG_C99)
# CHECK-NEXT: DW_AT_name [DW_FORM_strp]	( .debug_str[0x00000037] = "basic.c")
# CHECK-NEXT: DW_AT_stmt_list [DW_FORM_block4]
# CHECK-NEXT: DW_AT_comp_dir [DW_FORM_strp]	( .debug_str[0x0000003f] = "/Users/sgravani/Development/tests")
# CHECK-NEXT: DW_AT_low_pc [DW_FORM_addr]	(0x0000000000000000)
# CHECK-NEXT: DW_AT_high_pc [DW_FORM_data4]	(0x00000016){{[[:space:]]}}
# CHECK-NEXT: error: DIE has DW_AT_decl_file that references a file with index 1 and the compile unit has no line table{{[[:space:]]}}
# CHECK-NEXT: 0x0000002b: DW_TAG_subprogram [2] *
# CHECK-NEXT: DW_AT_low_pc [DW_FORM_addr]	(0x0000000000000000)
# CHECK-NEXT: DW_AT_high_pc [DW_FORM_data4]	(0x00000016)
# CHECK-NEXT: DW_AT_frame_base [DW_FORM_exprloc]	(DW_OP_reg6)
# CHECK-NEXT: DW_AT_name [DW_FORM_strp]	( .debug_str[0x00000061] = "main")
# CHECK-NEXT: DW_AT_decl_file [DW_FORM_data1]	(0x01)
# CHECK-NEXT: DW_AT_decl_line [DW_FORM_data1]	(1)
# CHECK-NEXT: DW_AT_prototyped [DW_FORM_flag_present]	(true)
# CHECK-NEXT: DW_AT_type [DW_FORM_ref4]	(cu + 0x0052 => {0x00000052})
# CHECK-NEXT: DW_AT_external [DW_FORM_flag_present]	(true){{[[:space:]]}}
# CHECK-NEXT: error: DIE has DW_AT_type with incompatible tag DW_TAG_null{{[[:space:]]}}
# CHECK-NEXT: 0x0000002b: DW_TAG_subprogram [2] *
# CHECK-NEXT: DW_AT_low_pc [DW_FORM_addr]       (0x0000000000000000)
# CHECK-NEXT: DW_AT_high_pc [DW_FORM_data4]     (0x00000016)
# CHECK-NEXT: DW_AT_frame_base [DW_FORM_exprloc]        (DW_OP_reg6)
# CHECK-NEXT: DW_AT_name [DW_FORM_strp] ( .debug_str[0x00000061] = "main")
# CHECK-NEXT: DW_AT_decl_file [DW_FORM_data1]   (0x01)
# CHECK-NEXT: DW_AT_decl_line [DW_FORM_data1]   (1)
# CHECK-NEXT: DW_AT_prototyped [DW_FORM_flag_present]   (true)
# CHECK-NEXT: DW_AT_type [DW_FORM_ref4] (cu + 0x0052 => {0x00000052})
# CHECK-NEXT: DW_AT_external [DW_FORM_flag_present]     (true){{[[:space:]]}}
# CHECK-NEXT: error: DIE has DW_AT_decl_file that references a file with index 1 and the compile unit has no line table{{[[:space:]]}}
# CHECK-NEXT: 0x00000044: DW_TAG_variable [3]
# CHECK-NEXT: DW_AT_location [DW_FORM_exprloc]	(DW_OP_fbreg -8)
# CHECK-NEXT: DW_AT_name [DW_FORM_strp]	( .debug_str[0x0000006a] = "a")
# CHECK-NEXT: DW_AT_decl_file [DW_FORM_data1]	(0x01)
# CHECK-NEXT: DW_AT_decl_line [DW_FORM_data1]	(2)
# CHECK-NEXT: DW_AT_use_location [DW_FORM_ref4]	(cu + 0x0053 => {0x00000053}){{[[:space:]]}}
# CHECK-NEXT: error: Compilation unit root DIE is not a unit DIE: DW_TAG_null.
# CHECK-NEXT: error: Compilation unit type (DW_UT_compile) and root DIE (DW_TAG_null) do not match.
# CHECK-NEXT: error: Units[2] - start offset: 0x00000068
# CHECK-NEXT: note: The length for this unit is too large for the .debug_info provided.
# CHECK-NEXT: note: The unit type encoding is not valid.


	.section	__TEXT,__text,regular,pure_instructions
	.globl	_main                   ## -- Begin function main
	.p2align	4, 0x90
_main:                                  ## @main
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
	xorl	%eax, %eax
	movl	$0, -4(%rbp)
Ltmp0:
	.loc	1 2 7 prologue_end      ## basic.c:2:7
	movl	$1, -8(%rbp)
	.loc	1 3 3                   ## basic.c:3:3
	popq	%rbp
	retq
Ltmp1:
Lfunc_end0:
	.cfi_endproc
                                        ## -- End function
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.asciz	"clang version 5.0.0 (trunk 308185) (llvm/trunk 308186)" ## string offset=0
	.asciz	"basic.c"               ## string offset=55
	.asciz	"/Users/sgravani/Development/tests" ## string offset=63
	.asciz	"main"                  ## string offset=97
	.asciz	"int"                   ## string offset=102
	.asciz	"a"                     ## string offset=106
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
	.byte	4                       ## DW_FORM_sec_offset -- error: DIE has invalid DW_AT_stmt_list encoding:
	.byte	27                      ## DW_AT_comp_dir
	.byte	14                      ## DW_FORM_strp
	.byte	17                      ## DW_AT_low_pc
	.byte	1                       ## DW_FORM_addr
	.byte	18                      ## DW_AT_high_pc
	.byte	6                       ## DW_FORM_data4
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	2                       ## Abbreviation Code
	.byte	46                      ## DW_TAG_subprogram
	.byte	1                       ## DW_CHILDREN_yes
	.byte	17                      ## DW_AT_low_pc
	.byte	1                       ## DW_FORM_addr
	.byte	18                      ## DW_AT_high_pc
	.byte	6                       ## DW_FORM_data4
	.byte	64                      ## DW_AT_frame_base
	.byte	24                      ## DW_FORM_exprloc
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	39                      ## DW_AT_prototyped
	.byte	25                      ## DW_FORM_flag_present
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
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
	.byte	14                      ## DW_FORM_strp
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	74                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	4                       ## Abbreviation Code
	.byte	36                      ## DW_TAG_base_type
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
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
	.long	87                      ## Length of Unit
	.short	5                       ## DWARF version number
	.byte	1                       ## DWARF Unit Type
	.byte	8                       ## Address Size (in bytes)
Lset0 = Lsection_abbrev-Lsection_abbrev ## Offset Into Abbrev. Section
	.long	Lset0
	.byte	1                       ## Abbrev [1] 0xc:0x4f DW_TAG_compile_unit
	.long	0                       ## DW_AT_producer
	.short	12                      ## DW_AT_language
	.long	55                      ## DW_AT_name
Lset1 = Lline_table_start0-Lsection_line ## DW_AT_stmt_list
	.long	Lset1
	.long	63                      ## DW_AT_comp_dir
	.quad	Lfunc_begin0            ## DW_AT_low_pc
Lset2 = Lfunc_end0-Lfunc_begin0         ## DW_AT_high_pc
	.long	Lset2
	.byte	2                       ## Abbrev [2] 0x2b:0x28 DW_TAG_subprogram
	.quad	Lfunc_begin0            ## DW_AT_low_pc
Lset3 = Lfunc_end0-Lfunc_begin0         ## DW_AT_high_pc
	.long	Lset3
	.byte	1                       ## DW_AT_frame_base
	.byte	86
	.long	97                      ## DW_AT_name
	.byte	1                       ## DW_AT_decl_file
	.byte	1                       ## DW_AT_decl_line
                                        ## DW_AT_prototyped
	.long	82                      ## DW_AT_type
                                        ## DW_AT_external
	.byte	3                       ## Abbrev [3] 0x44:0xe DW_TAG_variable
	.byte	2                       ## DW_AT_location
	.byte	145
	.byte	120
	.long	106                     ## DW_AT_name
	.byte	1                       ## DW_AT_decl_file
	.byte	2                       ## DW_AT_decl_line
	.long	83                      ## DW_AT_type
	.byte	0                       ## End Of Children Mark
	.byte	4                       ## Abbrev [4] 0x53:0x7 DW_TAG_base_type
	.long	102                     ## DW_AT_name
	.byte	5                       ## DW_AT_encoding
	.byte	4                       ## DW_AT_byte_size
	.byte	0                       ## End Of Children Mark
Lcu_begin1:
	.long	9                      ## Length of Unit
	.short	5                       ## DWARF version number
	.byte	1                       ## DWARF Unit Type
	.byte	4                       ## Address Size (in bytes)
	.long	0						## Abbrev offset
	.byte 	0
Ltu_begin0:
	.long	26                      ## Length of Unit -- Error: The length for this unit is too large for the .debug_info provided.
	.short	5                       ## DWARF version number
	.byte	0                       ## DWARF Unit Type
	.byte	4                       ## Address Size (in bytes)
	.long	0
	.quad	0
	.long   0
	.byte 	0

.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
