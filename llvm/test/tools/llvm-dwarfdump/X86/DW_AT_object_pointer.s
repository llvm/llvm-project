# RUN: llvm-mc %s -filetype obj -triple x86_64-apple-darwin -o %t.o
# RUN: llvm-dwarfdump -diff %t.o | FileCheck %s

# CHECK: DW_AT_type ("A*")
# CHECK: DW_AT_specification ("A")
# CHECK: DW_AT_object_pointer ()

	.section	__TEXT,__text,regular,pure_instructions
	.globl	__Z3fooi                ## -- Begin function _Z3fooi
	.p2align	4, 0x90
__Z3fooi:
Lfunc_begin0:
Ltmp0:
Ltmp1:
Lfunc_end0:
__ZN1AC1Ev:
Lfunc_begin1:
Ltmp2:
Ltmp3:
Lfunc_end1:
__ZN1AC2Ev:
Lfunc_begin2:
Ltmp4:
Ltmp5:
Lfunc_end2:
                                        ## -- End function
	.section	__DWARF,__debug_str,regular,debug
Linfo_string:
	.asciz	"clang version 3.2 (trunk 163586) (llvm/trunk 163570)" ## string offset=0
	.asciz	"bar.cpp"               ## string offset=53
	.asciz	"/Users/echristo/debug-tests" ## string offset=61
	.asciz	"foo"                   ## string offset=89
	.asciz	"_Z3fooi"               ## string offset=93
	.asciz	"A"                     ## string offset=101
	.asciz	"m_a"                   ## string offset=103
	.asciz	"int"                   ## string offset=107
	.asciz	"_ZN1AC1Ev"             ## string offset=111
	.asciz	"_ZN1AC2Ev"             ## string offset=121
	.asciz	"a"                     ## string offset=131
	.asciz	"this"                  ## string offset=133
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
	.byte	23                      ## DW_FORM_sec_offset
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
	.ascii	"\347\177"              ## DW_AT_APPLE_omit_frame_ptr
	.byte	25                      ## DW_FORM_flag_present
	.byte	64                      ## DW_AT_frame_base
	.byte	24                      ## DW_FORM_exprloc
	.byte	110                     ## DW_AT_linkage_name
	.byte	14                      ## DW_FORM_strp
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	63                      ## DW_AT_external
	.byte	25                      ## DW_FORM_flag_present
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	3                       ## Abbreviation Code
	.byte	5                       ## DW_TAG_formal_parameter
	.byte	0                       ## DW_CHILDREN_no
	.byte	2                       ## DW_AT_location
	.byte	24                      ## DW_FORM_exprloc
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	4                       ## Abbreviation Code
	.byte	11                      ## DW_TAG_lexical_block
	.byte	1                       ## DW_CHILDREN_yes
	.byte	17                      ## DW_AT_low_pc
	.byte	1                       ## DW_FORM_addr
	.byte	18                      ## DW_AT_high_pc
	.byte	6                       ## DW_FORM_data4
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	5                       ## Abbreviation Code
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
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	6                       ## Abbreviation Code
	.byte	2                       ## DW_TAG_class_type
	.byte	1                       ## DW_CHILDREN_yes
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	11                      ## DW_AT_byte_size
	.byte	11                      ## DW_FORM_data1
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.ascii	"\210\001"              ## DW_AT_alignment
	.byte	15                      ## DW_FORM_udata
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	7                       ## Abbreviation Code
	.byte	13                      ## DW_TAG_member
	.byte	0                       ## DW_CHILDREN_no
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.ascii	"\210\001"              ## DW_AT_alignment
	.byte	15                      ## DW_FORM_udata
	.byte	56                      ## DW_AT_data_member_location
	.byte	11                      ## DW_FORM_data1
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	8                       ## Abbreviation Code
	.byte	46                      ## DW_TAG_subprogram
	.byte	1                       ## DW_CHILDREN_yes
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	60                      ## DW_AT_declaration
	.byte	25                      ## DW_FORM_flag_present
	.byte	63                      ## DW_AT_external
	.byte	25                      ## DW_FORM_flag_present
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	9                       ## Abbreviation Code
	.byte	5                       ## DW_TAG_formal_parameter
	.byte	0                       ## DW_CHILDREN_no
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	52                      ## DW_AT_artificial
	.byte	25                      ## DW_FORM_flag_present
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	10                      ## Abbreviation Code
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
	.byte	11                      ## Abbreviation Code
	.byte	15                      ## DW_TAG_pointer_type
	.byte	0                       ## DW_CHILDREN_no
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	12                      ## Abbreviation Code
	.byte	46                      ## DW_TAG_subprogram
	.byte	1                       ## DW_CHILDREN_yes
	.byte	17                      ## DW_AT_low_pc
	.byte	1                       ## DW_FORM_addr
	.byte	18                      ## DW_AT_high_pc
	.byte	6                       ## DW_FORM_data4
	.ascii	"\347\177"              ## DW_AT_APPLE_omit_frame_ptr
	.byte	25                      ## DW_FORM_flag_present
	.byte	64                      ## DW_AT_frame_base
	.byte	24                      ## DW_FORM_exprloc
	.byte	100                     ## DW_AT_object_pointer
	.byte	19                      ## DW_FORM_ref4
	.byte	110                     ## DW_AT_linkage_name
	.byte	14                      ## DW_FORM_strp
	.byte	71                      ## DW_AT_specification
	.byte	19                      ## DW_FORM_ref4
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	13                      ## Abbreviation Code
	.byte	5                       ## DW_TAG_formal_parameter
	.byte	0                       ## DW_CHILDREN_no
	.byte	2                       ## DW_AT_location
	.byte	24                      ## DW_FORM_exprloc
	.byte	3                       ## DW_AT_name
	.byte	14                      ## DW_FORM_strp
	.byte	58                      ## DW_AT_decl_file
	.byte	11                      ## DW_FORM_data1
	.byte	59                      ## DW_AT_decl_line
	.byte	11                      ## DW_FORM_data1
	.byte	73                      ## DW_AT_type
	.byte	19                      ## DW_FORM_ref4
	.byte	52                      ## DW_AT_artificial
	.byte	25                      ## DW_FORM_flag_present
	.byte	0                       ## EOM(1)
	.byte	0                       ## EOM(2)
	.byte	0                       ## EOM(3)
	.section	__DWARF,__debug_info,regular,debug
Lsection_info:
Lcu_begin0:
	.long	244                     ## Length of Unit
	.short	4                       ## DWARF version number
.set Lset0, Lsection_abbrev-Lsection_abbrev ## Offset Into Abbrev. Section
	.long	Lset0
	.byte	8                       ## Address Size (in bytes)
	.byte	1                       ## Abbrev [1] 0xb:0xed DW_TAG_compile_unit
	.long	0                       ## DW_AT_producer
	.short	4                       ## DW_AT_language
	.long	53                      ## DW_AT_name
.set Lset1, Lline_table_start0-Lsection_line ## DW_AT_stmt_list
	.long	Lset1
	.long	61                      ## DW_AT_comp_dir
	.quad	Lfunc_begin0            ## DW_AT_low_pc
.set Lset2, Lfunc_end2-Lfunc_begin0     ## DW_AT_high_pc
	.long	Lset2
	.byte	2                       ## Abbrev [2] 0x2a:0x44 DW_TAG_subprogram
	.quad	Lfunc_begin0            ## DW_AT_low_pc
.set Lset3, Lfunc_end0-Lfunc_begin0     ## DW_AT_high_pc
	.long	Lset3
                                        ## DW_AT_APPLE_omit_frame_ptr
	.byte	1                       ## DW_AT_frame_base
	.byte	87
	.long	93                      ## DW_AT_linkage_name
	.long	89                      ## DW_AT_name
	.byte	1                       ## DW_AT_decl_file
	.byte	7                       ## DW_AT_decl_line
	.long	146                     ## DW_AT_type
                                        ## DW_AT_external
	.byte	3                       ## Abbrev [3] 0x47:0xa DW_TAG_formal_parameter
	.byte	2                       ## DW_AT_location
	.byte	145
	.byte	4
	.byte	1                       ## DW_AT_decl_file
	.byte	7                       ## DW_AT_decl_line
	.long	146                     ## DW_AT_type
	.byte	4                       ## Abbrev [4] 0x51:0x1c DW_TAG_lexical_block
	.quad	Ltmp0                   ## DW_AT_low_pc
.set Lset4, Ltmp1-Ltmp0                 ## DW_AT_high_pc
	.long	Lset4
	.byte	5                       ## Abbrev [5] 0x5e:0xe DW_TAG_variable
	.byte	2                       ## DW_AT_location
	.byte	145
	.byte	0
	.long	131                     ## DW_AT_name
	.byte	1                       ## DW_AT_decl_file
	.byte	8                       ## DW_AT_decl_line
	.long	110                     ## DW_AT_type
	.byte	0                       ## End Of Children Mark
	.byte	0                       ## End Of Children Mark
	.byte	6                       ## Abbrev [6] 0x6e:0x24 DW_TAG_class_type
	.long	101                     ## DW_AT_name
	.byte	4                       ## DW_AT_byte_size
	.byte	1                       ## DW_AT_decl_file
	.byte	1                       ## DW_AT_decl_line
	.byte	4                       ## DW_AT_alignment
	.byte	7                       ## Abbrev [7] 0x77:0xd DW_TAG_member
	.long	103                     ## DW_AT_name
	.long	146                     ## DW_AT_type
	.byte	1                       ## DW_AT_decl_file
	.byte	4                       ## DW_AT_decl_line
	.byte	4                       ## DW_AT_alignment
	.byte	0                       ## DW_AT_data_member_location
	.byte	8                       ## Abbrev [8] 0x84:0xd DW_TAG_subprogram
	.long	101                     ## DW_AT_name
	.byte	1                       ## DW_AT_decl_file
	.byte	3                       ## DW_AT_decl_line
                                        ## DW_AT_declaration
                                        ## DW_AT_external
	.byte	9                       ## Abbrev [9] 0x8b:0x5 DW_TAG_formal_parameter
	.long	153                     ## DW_AT_type
                                        ## DW_AT_artificial
	.byte	0                       ## End Of Children Mark
	.byte	0                       ## End Of Children Mark
	.byte	10                      ## Abbrev [10] 0x92:0x7 DW_TAG_base_type
	.long	107                     ## DW_AT_name
	.byte	5                       ## DW_AT_encoding
	.byte	4                       ## DW_AT_byte_size
	.byte	11                      ## Abbrev [11] 0x99:0x5 DW_TAG_pointer_type
	.long	110                     ## DW_AT_type
	.byte	12                      ## Abbrev [12] 0x9e:0x2a DW_TAG_subprogram
	.quad	Lfunc_begin1            ## DW_AT_low_pc
.set Lset5, Lfunc_end1-Lfunc_begin1     ## DW_AT_high_pc
	.long	Lset5
                                        ## DW_AT_APPLE_omit_frame_ptr
	.byte	1                       ## DW_AT_frame_base
	.byte	87
	.long	185                     ## DW_AT_object_pointer
	.long	111                     ## DW_AT_linkage_name
	.long	132                     ## DW_AT_specification
	.byte	13                      ## Abbrev [13] 0xb9:0xe DW_TAG_formal_parameter
	.byte	2                       ## DW_AT_location
	.byte	145
	.byte	0
	.long	133                     ## DW_AT_name
	.byte	1                       ## DW_AT_decl_file
	.byte	3                       ## DW_AT_decl_line
	.long	242                     ## DW_AT_type
                                        ## DW_AT_artificial
	.byte	0                       ## End Of Children Mark
	.byte	12                      ## Abbrev [12] 0xc8:0x2a DW_TAG_subprogram
	.quad	Lfunc_begin2            ## DW_AT_low_pc
.set Lset6, Lfunc_end2-Lfunc_begin2     ## DW_AT_high_pc
	.long	Lset6
                                        ## DW_AT_APPLE_omit_frame_ptr
	.byte	1                       ## DW_AT_frame_base
	.byte	87
	.long	227                     ## DW_AT_object_pointer
	.long	121                     ## DW_AT_linkage_name
	.long	132                     ## DW_AT_specification
	.byte	13                      ## Abbrev [13] 0xe3:0xe DW_TAG_formal_parameter
	.byte	2                       ## DW_AT_location
	.byte	145
	.byte	120
	.long	133                     ## DW_AT_name
	.byte	1                       ## DW_AT_decl_file
	.byte	3                       ## DW_AT_decl_line
	.long	242                     ## DW_AT_type
                                        ## DW_AT_artificial
	.byte	0                       ## End Of Children Mark
	.byte	11                      ## Abbrev [11] 0xf2:0x5 DW_TAG_pointer_type
	.long	110                     ## DW_AT_type
	.byte	0                       ## End Of Children Mark
	.section	__DWARF,__debug_macinfo,regular,debug
Ldebug_macinfo:
	.byte	0                       ## End Of Macro List Mark
	.section	__DWARF,__apple_names,regular,debug
Lnames_begin:
	.long	1212240712              ## Header Magic
	.short	1                       ## Header Version
	.short	0                       ## Header Hash Function
	.long	5                       ## Header Bucket Count
	.long	5                       ## Header Hash Count
	.long	12                      ## Header Data Length
	.long	0                       ## HeaderData Die Offset Base
	.long	1                       ## HeaderData Atom Count
	.short	1                       ## DW_ATOM_die_offset
	.short	6                       ## DW_FORM_data4
	.long	0                       ## Bucket 0
	.long	2                       ## Bucket 1
	.long	-1                      ## Bucket 2
	.long	3                       ## Bucket 3
	.long	4                       ## Bucket 4
	.long	649621230               ## Hash in Bucket 0
	.long	1784752350              ## Hash in Bucket 0
	.long	649620141               ## Hash in Bucket 1
	.long	177638                  ## Hash in Bucket 3
	.long	193491849               ## Hash in Bucket 4
.set Lset7, LNames4-Lnames_begin        ## Offset in Bucket 0
	.long	Lset7
.set Lset8, LNames3-Lnames_begin        ## Offset in Bucket 0
	.long	Lset8
.set Lset9, LNames2-Lnames_begin        ## Offset in Bucket 1
	.long	Lset9
.set Lset10, LNames0-Lnames_begin       ## Offset in Bucket 3
	.long	Lset10
.set Lset11, LNames1-Lnames_begin       ## Offset in Bucket 4
	.long	Lset11
LNames4:
	.long	121                     ## _ZN1AC2Ev
	.long	1                       ## Num DIEs
	.long	200
	.long	0
LNames3:
	.long	93                      ## _Z3fooi
	.long	1                       ## Num DIEs
	.long	42
	.long	0
LNames2:
	.long	111                     ## _ZN1AC1Ev
	.long	1                       ## Num DIEs
	.long	158
	.long	0
LNames0:
	.long	101                     ## A
	.long	2                       ## Num DIEs
	.long	158
	.long	200
	.long	0
LNames1:
	.long	89                      ## foo
	.long	1                       ## Num DIEs
	.long	42
	.long	0
	.section	__DWARF,__apple_objc,regular,debug
Lobjc_begin:
	.long	1212240712              ## Header Magic
	.short	1                       ## Header Version
	.short	0                       ## Header Hash Function
	.long	1                       ## Header Bucket Count
	.long	0                       ## Header Hash Count
	.long	12                      ## Header Data Length
	.long	0                       ## HeaderData Die Offset Base
	.long	1                       ## HeaderData Atom Count
	.short	1                       ## DW_ATOM_die_offset
	.short	6                       ## DW_FORM_data4
	.long	-1                      ## Bucket 0
	.section	__DWARF,__apple_namespac,regular,debug
Lnamespac_begin:
	.long	1212240712              ## Header Magic
	.short	1                       ## Header Version
	.short	0                       ## Header Hash Function
	.long	1                       ## Header Bucket Count
	.long	0                       ## Header Hash Count
	.long	12                      ## Header Data Length
	.long	0                       ## HeaderData Die Offset Base
	.long	1                       ## HeaderData Atom Count
	.short	1                       ## DW_ATOM_die_offset
	.short	6                       ## DW_FORM_data4
	.long	-1                      ## Bucket 0
	.section	__DWARF,__apple_types,regular,debug
Ltypes_begin:
	.long	1212240712              ## Header Magic
	.short	1                       ## Header Version
	.short	0                       ## Header Hash Function
	.long	2                       ## Header Bucket Count
	.long	2                       ## Header Hash Count
	.long	20                      ## Header Data Length
	.long	0                       ## HeaderData Die Offset Base
	.long	3                       ## HeaderData Atom Count
	.short	1                       ## DW_ATOM_die_offset
	.short	6                       ## DW_FORM_data4
	.short	3                       ## DW_ATOM_die_tag
	.short	5                       ## DW_FORM_data2
	.short	4                       ## DW_ATOM_type_flags
	.short	11                      ## DW_FORM_data1
	.long	0                       ## Bucket 0
	.long	-1                      ## Bucket 1
	.long	177638                  ## Hash in Bucket 0
	.long	193495088               ## Hash in Bucket 0
.set Lset12, Ltypes0-Ltypes_begin       ## Offset in Bucket 0
	.long	Lset12
.set Lset13, Ltypes1-Ltypes_begin       ## Offset in Bucket 0
	.long	Lset13
Ltypes0:
	.long	101                     ## A
	.long	1                       ## Num DIEs
	.long	110
	.short	2
	.byte	0
	.long	0
Ltypes1:
	.long	107                     ## int
	.long	1                       ## Num DIEs
	.long	146
	.short	36
	.byte	0
	.long	0

.subsections_via_symbols
	.section	__DWARF,__debug_line,regular,debug
Lsection_line:
Lline_table_start0:
