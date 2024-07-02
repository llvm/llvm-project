	.text
	.file	"pr-43860.cpp"
	.globaltype	__stack_pointer, i32
	.functype	_Z4testii (i32, i32) -> (i32)
	.section	.text._Z4testii,"",@
	.hidden	_Z4testii                       # -- Begin function _Z4testii
	.globl	_Z4testii
	.type	_Z4testii,@function
_Z4testii:                              # @_Z4testii
.Lfunc_begin0:
	.file	1 "/data/projects/scripts/regression-suite/input/general" "pr-43860.cpp"
	.loc	1 11 0                          # pr-43860.cpp:11:0
	.functype	_Z4testii (i32, i32) -> (i32)
	.local  	i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
# %bb.0:                                # %entry
	global.get	__stack_pointer
	local.set	2
	i32.const	32
	local.set	3
	local.get	2
	local.get	3
	i32.sub 
	local.set	4
	local.get	4
	local.get	0
	i32.store	16
	local.get	4
	local.get	1
	i32.store	12
.Ltmp0:
	.loc	1 12 11 prologue_end            # pr-43860.cpp:12:11
	local.get	4
	i32.load	16
	local.set	5
	.loc	1 12 7 is_stmt 0                # pr-43860.cpp:12:7
	local.get	4
	local.get	5
	i32.store	8
	.loc	1 13 23 is_stmt 1               # pr-43860.cpp:13:23
	local.get	4
	i32.load	12
	local.set	6
	local.get	4
	local.get	6
	i32.store	28
.Ltmp1:
	.loc	1 3 15                          # pr-43860.cpp:3:15
	local.get	4
	i32.load	28
	local.set	7
	.loc	1 3 7 is_stmt 0                 # pr-43860.cpp:3:7
	local.get	4
	local.get	7
	i32.store	24
.Ltmp2:
	.loc	1 5 17 is_stmt 1                # pr-43860.cpp:5:17
	local.get	4
	i32.load	28
	local.set	8
	.loc	1 5 25 is_stmt 0                # pr-43860.cpp:5:25
	local.get	4
	i32.load	24
	local.set	9
	.loc	1 5 23                          # pr-43860.cpp:5:23
	local.get	8
	local.get	9
	i32.add 
	local.set	10
	.loc	1 5 9                           # pr-43860.cpp:5:9
	local.get	4
	local.get	10
	i32.store	20
	.loc	1 6 13 is_stmt 1                # pr-43860.cpp:6:13
	local.get	4
	i32.load	20
	local.set	11
	.loc	1 6 11 is_stmt 0                # pr-43860.cpp:6:11
	local.get	4
	local.get	11
	i32.store	24
.Ltmp3:
	.loc	1 8 10 is_stmt 1                # pr-43860.cpp:8:10
	local.get	4
	i32.load	24
	local.set	12
.Ltmp4:
	.loc	1 13 5                          # pr-43860.cpp:13:5
	local.get	4
	i32.load	8
	local.set	13
	local.get	13
	local.get	12
	i32.add 
	local.set	14
	local.get	4
	local.get	14
	i32.store	8
	.loc	1 14 10                         # pr-43860.cpp:14:10
	local.get	4
	i32.load	8
	local.set	15
	.loc	1 14 3 is_stmt 0                # pr-43860.cpp:14:3
	local.get	15
	return
	end_function
.Ltmp5:
.Lfunc_end0:
                                        # -- End function
	.section	.debug_abbrev,"",@
	.int8	1                               # Abbreviation Code
	.int8	17                              # DW_TAG_compile_unit
	.int8	1                               # DW_CHILDREN_yes
	.int8	37                              # DW_AT_producer
	.int8	14                              # DW_FORM_strp
	.int8	19                              # DW_AT_language
	.int8	5                               # DW_FORM_data2
	.int8	3                               # DW_AT_name
	.int8	14                              # DW_FORM_strp
	.int8	16                              # DW_AT_stmt_list
	.int8	23                              # DW_FORM_sec_offset
	.int8	27                              # DW_AT_comp_dir
	.int8	14                              # DW_FORM_strp
	.int8	17                              # DW_AT_low_pc
	.int8	1                               # DW_FORM_addr
	.int8	18                              # DW_AT_high_pc
	.int8	6                               # DW_FORM_data4
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	2                               # Abbreviation Code
	.int8	46                              # DW_TAG_subprogram
	.int8	1                               # DW_CHILDREN_yes
	.int8	110                             # DW_AT_linkage_name
	.int8	14                              # DW_FORM_strp
	.int8	3                               # DW_AT_name
	.int8	14                              # DW_FORM_strp
	.int8	58                              # DW_AT_decl_file
	.int8	11                              # DW_FORM_data1
	.int8	59                              # DW_AT_decl_line
	.int8	11                              # DW_FORM_data1
	.int8	73                              # DW_AT_type
	.int8	19                              # DW_FORM_ref4
	.int8	63                              # DW_AT_external
	.int8	25                              # DW_FORM_flag_present
	.int8	32                              # DW_AT_inline
	.int8	11                              # DW_FORM_data1
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	3                               # Abbreviation Code
	.int8	5                               # DW_TAG_formal_parameter
	.int8	0                               # DW_CHILDREN_no
	.int8	3                               # DW_AT_name
	.int8	14                              # DW_FORM_strp
	.int8	58                              # DW_AT_decl_file
	.int8	11                              # DW_FORM_data1
	.int8	59                              # DW_AT_decl_line
	.int8	11                              # DW_FORM_data1
	.int8	73                              # DW_AT_type
	.int8	19                              # DW_FORM_ref4
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	4                               # Abbreviation Code
	.int8	52                              # DW_TAG_variable
	.int8	0                               # DW_CHILDREN_no
	.int8	3                               # DW_AT_name
	.int8	14                              # DW_FORM_strp
	.int8	58                              # DW_AT_decl_file
	.int8	11                              # DW_FORM_data1
	.int8	59                              # DW_AT_decl_line
	.int8	11                              # DW_FORM_data1
	.int8	73                              # DW_AT_type
	.int8	19                              # DW_FORM_ref4
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	5                               # Abbreviation Code
	.int8	11                              # DW_TAG_lexical_block
	.int8	1                               # DW_CHILDREN_yes
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	6                               # Abbreviation Code
	.int8	36                              # DW_TAG_base_type
	.int8	0                               # DW_CHILDREN_no
	.int8	3                               # DW_AT_name
	.int8	14                              # DW_FORM_strp
	.int8	62                              # DW_AT_encoding
	.int8	11                              # DW_FORM_data1
	.int8	11                              # DW_AT_byte_size
	.int8	11                              # DW_FORM_data1
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	7                               # Abbreviation Code
	.int8	46                              # DW_TAG_subprogram
	.int8	1                               # DW_CHILDREN_yes
	.int8	17                              # DW_AT_low_pc
	.int8	1                               # DW_FORM_addr
	.int8	18                              # DW_AT_high_pc
	.int8	6                               # DW_FORM_data4
	.int8	64                              # DW_AT_frame_base
	.int8	24                              # DW_FORM_exprloc
	.int8	110                             # DW_AT_linkage_name
	.int8	14                              # DW_FORM_strp
	.int8	3                               # DW_AT_name
	.int8	14                              # DW_FORM_strp
	.int8	58                              # DW_AT_decl_file
	.int8	11                              # DW_FORM_data1
	.int8	59                              # DW_AT_decl_line
	.int8	11                              # DW_FORM_data1
	.int8	73                              # DW_AT_type
	.int8	19                              # DW_FORM_ref4
	.int8	63                              # DW_AT_external
	.int8	25                              # DW_FORM_flag_present
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	8                               # Abbreviation Code
	.int8	5                               # DW_TAG_formal_parameter
	.int8	0                               # DW_CHILDREN_no
	.int8	2                               # DW_AT_location
	.int8	24                              # DW_FORM_exprloc
	.int8	3                               # DW_AT_name
	.int8	14                              # DW_FORM_strp
	.int8	58                              # DW_AT_decl_file
	.int8	11                              # DW_FORM_data1
	.int8	59                              # DW_AT_decl_line
	.int8	11                              # DW_FORM_data1
	.int8	73                              # DW_AT_type
	.int8	19                              # DW_FORM_ref4
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	9                               # Abbreviation Code
	.int8	52                              # DW_TAG_variable
	.int8	0                               # DW_CHILDREN_no
	.int8	2                               # DW_AT_location
	.int8	24                              # DW_FORM_exprloc
	.int8	3                               # DW_AT_name
	.int8	14                              # DW_FORM_strp
	.int8	58                              # DW_AT_decl_file
	.int8	11                              # DW_FORM_data1
	.int8	59                              # DW_AT_decl_line
	.int8	11                              # DW_FORM_data1
	.int8	73                              # DW_AT_type
	.int8	19                              # DW_FORM_ref4
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	10                              # Abbreviation Code
	.int8	29                              # DW_TAG_inlined_subroutine
	.int8	1                               # DW_CHILDREN_yes
	.int8	49                              # DW_AT_abstract_origin
	.int8	19                              # DW_FORM_ref4
	.int8	17                              # DW_AT_low_pc
	.int8	1                               # DW_FORM_addr
	.int8	18                              # DW_AT_high_pc
	.int8	6                               # DW_FORM_data4
	.int8	88                              # DW_AT_call_file
	.int8	11                              # DW_FORM_data1
	.int8	89                              # DW_AT_call_line
	.int8	11                              # DW_FORM_data1
	.int8	87                              # DW_AT_call_column
	.int8	11                              # DW_FORM_data1
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	11                              # Abbreviation Code
	.int8	5                               # DW_TAG_formal_parameter
	.int8	0                               # DW_CHILDREN_no
	.int8	2                               # DW_AT_location
	.int8	24                              # DW_FORM_exprloc
	.int8	49                              # DW_AT_abstract_origin
	.int8	19                              # DW_FORM_ref4
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	12                              # Abbreviation Code
	.int8	52                              # DW_TAG_variable
	.int8	0                               # DW_CHILDREN_no
	.int8	2                               # DW_AT_location
	.int8	24                              # DW_FORM_exprloc
	.int8	49                              # DW_AT_abstract_origin
	.int8	19                              # DW_FORM_ref4
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	13                              # Abbreviation Code
	.int8	11                              # DW_TAG_lexical_block
	.int8	1                               # DW_CHILDREN_yes
	.int8	17                              # DW_AT_low_pc
	.int8	1                               # DW_FORM_addr
	.int8	18                              # DW_AT_high_pc
	.int8	6                               # DW_FORM_data4
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	0                               # EOM(3)
	.section	.debug_info,"",@
.Lcu_begin0:
	.int32	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.int16	4                               # DWARF version number
	.int32	.debug_abbrev0                  # Offset Into Abbrev. Section
	.int8	4                               # Address Size (in bytes)
	.int8	1                               # Abbrev [1] 0xb:0xd1 DW_TAG_compile_unit
	.int32	.Linfo_string0                  # DW_AT_producer
	.int16	33                              # DW_AT_language
	.int32	.Linfo_string1                  # DW_AT_name
	.int32	.Lline_table_start0             # DW_AT_stmt_list
	.int32	.Linfo_string2                  # DW_AT_comp_dir
	.int32	.Lfunc_begin0                   # DW_AT_low_pc
	.int32	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.int8	2                               # Abbrev [2] 0x26:0x34 DW_TAG_subprogram
	.int32	.Linfo_string3                  # DW_AT_linkage_name
	.int32	.Linfo_string4                  # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	2                               # DW_AT_decl_line
	.int32	90                              # DW_AT_type
                                        # DW_AT_external
	.int8	1                               # DW_AT_inline
	.int8	3                               # Abbrev [3] 0x36:0xb DW_TAG_formal_parameter
	.int32	.Linfo_string6                  # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	2                               # DW_AT_decl_line
	.int32	90                              # DW_AT_type
	.int8	4                               # Abbrev [4] 0x41:0xb DW_TAG_variable
	.int32	.Linfo_string7                  # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	3                               # DW_AT_decl_line
	.int32	90                              # DW_AT_type
	.int8	5                               # Abbrev [5] 0x4c:0xd DW_TAG_lexical_block
	.int8	4                               # Abbrev [4] 0x4d:0xb DW_TAG_variable
	.int32	.Linfo_string8                  # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	5                               # DW_AT_decl_line
	.int32	90                              # DW_AT_type
	.int8	0                               # End Of Children Mark
	.int8	0                               # End Of Children Mark
	.int8	6                               # Abbrev [6] 0x5a:0x7 DW_TAG_base_type
	.int32	.Linfo_string5                  # DW_AT_name
	.int8	5                               # DW_AT_encoding
	.int8	4                               # DW_AT_byte_size
	.int8	7                               # Abbrev [7] 0x61:0x7a DW_TAG_subprogram
	.int32	.Lfunc_begin0                   # DW_AT_low_pc
	.int32	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.int8	4                               # DW_AT_frame_base
	.int8	237
	.int8	0
	.int8	4
	.int8	159
	.int32	.Linfo_string9                  # DW_AT_linkage_name
	.int32	.Linfo_string10                 # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	11                              # DW_AT_decl_line
	.int32	90                              # DW_AT_type
                                        # DW_AT_external
	.int8	8                               # Abbrev [8] 0x7d:0xe DW_TAG_formal_parameter
	.int8	2                               # DW_AT_location
	.int8	145
	.int8	16
	.int32	.Linfo_string11                 # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	11                              # DW_AT_decl_line
	.int32	90                              # DW_AT_type
	.int8	8                               # Abbrev [8] 0x8b:0xe DW_TAG_formal_parameter
	.int8	2                               # DW_AT_location
	.int8	145
	.int8	12
	.int32	.Linfo_string12                 # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	11                              # DW_AT_decl_line
	.int32	90                              # DW_AT_type
	.int8	9                               # Abbrev [9] 0x99:0xe DW_TAG_variable
	.int8	2                               # DW_AT_location
	.int8	145
	.int8	8
	.int32	.Linfo_string13                 # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	12                              # DW_AT_decl_line
	.int32	90                              # DW_AT_type
	.int8	10                              # Abbrev [10] 0xa7:0x33 DW_TAG_inlined_subroutine
	.int32	38                              # DW_AT_abstract_origin
	.int32	.Ltmp1                          # DW_AT_low_pc
	.int32	.Ltmp4-.Ltmp1                   # DW_AT_high_pc
	.int8	1                               # DW_AT_call_file
	.int8	13                              # DW_AT_call_line
	.int8	8                               # DW_AT_call_column
	.int8	11                              # Abbrev [11] 0xb7:0x8 DW_TAG_formal_parameter
	.int8	2                               # DW_AT_location
	.int8	145
	.int8	28
	.int32	54                              # DW_AT_abstract_origin
	.int8	12                              # Abbrev [12] 0xbf:0x8 DW_TAG_variable
	.int8	2                               # DW_AT_location
	.int8	145
	.int8	24
	.int32	65                              # DW_AT_abstract_origin
	.int8	13                              # Abbrev [13] 0xc7:0x12 DW_TAG_lexical_block
	.int32	.Ltmp2                          # DW_AT_low_pc
	.int32	.Ltmp3-.Ltmp2                   # DW_AT_high_pc
	.int8	12                              # Abbrev [12] 0xd0:0x8 DW_TAG_variable
	.int8	2                               # DW_AT_location
	.int8	145
	.int8	20
	.int32	77                              # DW_AT_abstract_origin
	.int8	0                               # End Of Children Mark
	.int8	0                               # End Of Children Mark
	.int8	0                               # End Of Children Mark
	.int8	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"S",@
.Linfo_string0:
	.asciz	"clang version 19.0.0git (/data/projects/llvm-root/llvm-project/clang 2db6703f0c257d293df455e2dff8c1fb695c4100)" # string offset=0
.Linfo_string1:
	.asciz	"pr-43860.cpp"                  # string offset=111
.Linfo_string2:
	.asciz	"/data/projects/scripts/regression-suite/input/general" # string offset=124
.Linfo_string3:
	.asciz	"_Z14InlineFunctioni"           # string offset=178
.Linfo_string4:
	.asciz	"InlineFunction"                # string offset=198
.Linfo_string5:
	.asciz	"int"                           # string offset=213
.Linfo_string6:
	.asciz	"Param"                         # string offset=217
.Linfo_string7:
	.asciz	"Var_1"                         # string offset=223
.Linfo_string8:
	.asciz	"Var_2"                         # string offset=229
.Linfo_string9:
	.asciz	"_Z4testii"                     # string offset=235
.Linfo_string10:
	.asciz	"test"                          # string offset=245
.Linfo_string11:
	.asciz	"Param_1"                       # string offset=250
.Linfo_string12:
	.asciz	"Param_2"                       # string offset=258
.Linfo_string13:
	.asciz	"A"                             # string offset=266
	.ident	"clang version 19.0.0git (/data/projects/llvm-root/llvm-project/clang 2db6703f0c257d293df455e2dff8c1fb695c4100)"
	.section	.custom_section.producers,"",@
	.int8	2
	.int8	8
	.ascii	"language"
	.int8	1
	.int8	14
	.ascii	"C_plus_plus_14"
	.int8	0
	.int8	12
	.ascii	"processed-by"
	.int8	1
	.int8	5
	.ascii	"clang"
	.int8	96
	.ascii	"19.0.0git (/data/projects/llvm-root/llvm-project/clang 2db6703f0c257d293df455e2dff8c1fb695c4100)"
	.section	.debug_str,"S",@
	.section	.custom_section.target_features,"",@
	.int8	2
	.int8	43
	.int8	15
	.ascii	"mutable-globals"
	.int8	43
	.int8	8
	.ascii	"sign-ext"
	.section	.debug_str,"S",@
	.section	.debug_line,"",@
.Lline_table_start0:
