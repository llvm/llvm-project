# RUN: llvm-mc < %s -triple=s390x-linux-gnu -filetype=obj | llvm-dwarfdump - | FileCheck %s
#
# We use both R_390_32 and R_390_64 to encode the dwarf information.
# Test that they are used correctly.  This uses the assembly output
# for variable-loc.ll
#
# A couple of R_390_32s, both at 0 and elsewhere:
#
# CHECK: DW_AT_producer [DW_FORM_strp] ( .debug_str[0x00000000] = "clang version 3.2 ")
# CHECK: DW_AT_name [DW_FORM_strp] ( .debug_str[0x00000013] = "simple.c")
#
# A couple of R_390_64s similarly:
#
# CHECK: DW_AT_low_pc [DW_FORM_addr] (0x0000000000000000)
# CHECK: DW_AT_high_pc [DW_FORM_addr] (0x0000000000000050)


	.file	"test/DebugInfo/SystemZ/variable-loc.ll"
	.section	.debug_info,"",@progbits
.Lsection_info:
	.section	.debug_abbrev,"",@progbits
.Lsection_abbrev:
	.section	.debug_aranges,"",@progbits
	.section	.debug_macinfo,"",@progbits
	.section	.debug_line,"",@progbits
.Lsection_line:
	.section	.debug_loc,"",@progbits
	.section	.debug_pubtypes,"",@progbits
	.section	.debug_str,"MS",@progbits,1
.Linfo_string:
	.section	.debug_ranges,"",@progbits
.Ldebug_range:
	.section	.debug_loc,"",@progbits
.Lsection_debug_loc:
	.text
.Ltext_begin:
	.data
	.file	1 "simple.c"
	.file	2 "<stdin>"
	.text
	.globl	main
	.align	4
	.type	main,@function
main:                                   # @main
	.cfi_startproc
.Lfunc_begin0:
	.loc	2 18 0                  # :18:0
# BB#0:                                 # %entry
	stmg	%r12, %r15, 96(%r15)
.Ltmp2:
	.cfi_offset %r12, -64
.Ltmp3:
	.cfi_offset %r13, -56
.Ltmp4:
	.cfi_offset %r14, -48
.Ltmp5:
	.cfi_offset %r15, -40
	aghi	%r15, -568
.Ltmp6:
	.cfi_def_cfa_offset 728
	mvhi	564(%r15), 0
	la	%r13, 164(%r15)
	lhi	%r12, 100
	.loc	2 22 3 prologue_end     # :22:3
.Ltmp7:
	lgr	%r2, %r13
	lr	%r3, %r12
	brasl	%r14, populate_array@PLT
	.loc	2 23 9                  # :23:9
	lgr	%r2, %r13
	lr	%r3, %r12
	brasl	%r14, sum_array@PLT
	lr	%r0, %r2
	st	%r0, 160(%r15)
	.loc	2 24 3                  # :24:3
	larl	%r2, .L.str
	lr	%r3, %r0
	brasl	%r14, printf@PLT
	lhi	%r2, 0
	.loc	2 26 3                  # :26:3
	lmg	%r12, %r15, 664(%r15)
	br	%r14
.Ltmp8:
.Ltmp9:
	.size	main, .Ltmp9-main
.Lfunc_end0:
	.cfi_endproc

	.type	.L.str,@object          # @.str
	.section	.rodata.str1.2,"aMS",@progbits,1
	.align	2
.L.str:
	.asciz	 "Total is %d\n"
	.size	.L.str, 13

	.cfi_sections .debug_frame
	.text
.Ltext_end:
	.data
.Ldata_end:
	.text
.Lsection_end1:
	.section	.debug_info,"",@progbits
.L.debug_info_begin0:
	.long	155                     # Length of Compilation Unit Info
	.short	2                       # DWARF version number
	.long	.L.debug_abbrev_begin   # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] 0xb:0x94 DW_TAG_compile_unit
	.long	.Linfo_string0          # DW_AT_producer
	.short	12                      # DW_AT_language
	.long	.Linfo_string1          # DW_AT_name
	.quad	0                       # DW_AT_low_pc
	.long	.Lsection_line          # DW_AT_stmt_list
	.long	.Linfo_string2          # DW_AT_comp_dir
	.byte	2                       # Abbrev [2] 0x26:0x7 DW_TAG_subprogram
	.long	.Linfo_string3          # DW_AT_name
	.byte	2                       # DW_AT_decl_file
	.byte	4                       # DW_AT_decl_line
                                        # DW_AT_prototyped
                                        # DW_AT_external
	.byte	3                       # Abbrev [3] 0x2d:0xb DW_TAG_subprogram
	.long	.Linfo_string4          # DW_AT_name
	.byte	2                       # DW_AT_decl_file
	.byte	9                       # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	56                      # DW_AT_type
                                        # DW_AT_external
	.byte	4                       # Abbrev [4] 0x38:0x7 DW_TAG_base_type
	.long	.Linfo_string5          # DW_AT_name
	.byte	5                       # DW_AT_encoding
	.byte	4                       # DW_AT_byte_size
	.byte	5                       # Abbrev [5] 0x3f:0xb DW_TAG_subprogram
	.long	.Linfo_string6          # DW_AT_name
	.byte	2                       # DW_AT_decl_file
	.byte	18                      # DW_AT_decl_line
                                        # DW_AT_prototyped
	.long	56                      # DW_AT_type
                                        # DW_AT_external
                                        # DW_AT_declaration
	.byte	6                       # Abbrev [6] 0x4a:0x7 DW_TAG_base_type
	.long	.Linfo_string5          # DW_AT_name
	.byte	4                       # DW_AT_byte_size
	.byte	5                       # DW_AT_encoding
	.byte	7                       # Abbrev [7] 0x51:0x5 DW_TAG_array_type
	.long	56                      # DW_AT_type
	.byte	8                       # Abbrev [8] 0x56:0x48 DW_TAG_subprogram
	.long	63                      # DW_AT_specification
	.quad	.Lfunc_begin0           # DW_AT_low_pc
	.quad	.Lfunc_end0             # DW_AT_high_pc
	.byte	1                       # DW_AT_frame_base
	.byte	95
                                        # DW_AT_APPLE_omit_frame_ptr
	.byte	9                       # Abbrev [9] 0x6d:0x30 DW_TAG_lexical_block
	.quad	.Ltmp7                  # DW_AT_low_pc
	.quad	.Ltmp8                  # DW_AT_high_pc
	.byte	10                      # Abbrev [10] 0x7e:0xf DW_TAG_variable
	.long	.Linfo_string7          # DW_AT_name
	.byte	2                       # DW_AT_decl_file
	.byte	19                      # DW_AT_decl_line
	.long	81                      # DW_AT_type
	.byte	3                       # DW_AT_location
	.byte	145
	.ascii	 "\244\001"
	.byte	10                      # Abbrev [10] 0x8d:0xf DW_TAG_variable
	.long	.Linfo_string8          # DW_AT_name
	.byte	2                       # DW_AT_decl_file
	.byte	20                      # DW_AT_decl_line
	.long	56                      # DW_AT_type
	.byte	3                       # DW_AT_location
	.byte	145
	.ascii	 "\240\001"
	.byte	0                       # End Of Children Mark
	.byte	0                       # End Of Children Mark
	.byte	0                       # End Of Children Mark
.L.debug_info_end0:
	.section	.debug_abbrev,"",@progbits
.L.debug_abbrev_begin:
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	37                      # DW_AT_producer
	.byte	14                      # DW_FORM_strp
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	16                      # DW_AT_stmt_list
	.byte	6                       # DW_FORM_data4
	.byte	27                      # DW_AT_comp_dir
	.byte	14                      # DW_FORM_strp
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	2                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	14                      # DW_FORM_strp
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	39                      # DW_AT_prototyped
	.byte	25                      # DW_FORM_flag_present
	.byte	63                      # DW_AT_external
	.byte	25                      # DW_FORM_flag_present
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	3                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	0                       # DW_CHILDREN_no
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
	.byte	4                       # Abbreviation Code
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
	.byte	5                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	0                       # DW_CHILDREN_no
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
	.byte	60                      # DW_AT_declaration
	.byte	25                      # DW_FORM_flag_present
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	6                       # Abbreviation Code
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
	.byte	7                       # Abbreviation Code
	.byte	1                       # DW_TAG_array_type
	.byte	0                       # DW_CHILDREN_no
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	8                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	1                       # DW_CHILDREN_yes
	.byte	71                      # DW_AT_specification
	.byte	19                      # DW_FORM_ref4
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	1                       # DW_FORM_addr
	.byte	64                      # DW_AT_frame_base
	.byte	10                      # DW_FORM_block1
	.ascii	 "\347\177"             # DW_AT_APPLE_omit_frame_ptr
	.byte	25                      # DW_FORM_flag_present
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	9                       # Abbreviation Code
	.byte	11                      # DW_TAG_lexical_block
	.byte	1                       # DW_CHILDREN_yes
	.byte	17                      # DW_AT_low_pc
	.byte	1                       # DW_FORM_addr
	.byte	18                      # DW_AT_high_pc
	.byte	1                       # DW_FORM_addr
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	10                      # Abbreviation Code
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
	.byte	2                       # DW_AT_location
	.byte	10                      # DW_FORM_block1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
.L.debug_abbrev_end:
	.section	.debug_aranges,"",@progbits
	.section	.debug_ranges,"",@progbits
	.section	.debug_macinfo,"",@progbits
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	 "clang version 3.2 "
.Linfo_string1:
	.asciz	 "simple.c"
.Linfo_string2:
	.asciz	 "/home/timnor01/a64-trunk/build"
.Linfo_string3:
	.asciz	 "populate_array"
.Linfo_string4:
	.asciz	 "sum_array"
.Linfo_string5:
	.asciz	 "int"
.Linfo_string6:
	.asciz	 "main"
.Linfo_string7:
	.asciz	 "main_arr"
.Linfo_string8:
	.asciz	 "val"

	.section	".note.GNU-stack","",@progbits
