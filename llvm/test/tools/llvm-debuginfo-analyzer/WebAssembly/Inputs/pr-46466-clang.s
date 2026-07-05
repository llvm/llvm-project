	.text
	.file	"pr-46466.cpp"
	.file	1 "/data/projects/scripts/regression-suite/input/general" "pr-46466.cpp"
	.functype	_Z4testv () -> (i32)
	.section	.text._Z4testv,"",@
	.hidden	_Z4testv                        # -- Begin function _Z4testv
	.globl	_Z4testv
	.type	_Z4testv,@function
_Z4testv:                               # @_Z4testv
.Lfunc_begin0:
	.functype	_Z4testv () -> (i32)
	.local  	i32
# %bb.0:                                # %entry
	.loc	1 10 3 prologue_end             # pr-46466.cpp:10:3
	i32.const	1
	local.set	0
	local.get	0
	return
	end_function
.Ltmp0:
.Lfunc_end0:
                                        # -- End function
	.hidden	S                               # @S
	.type	S,@object
	.section	.bss.S,"",@
	.globl	S
S:
	.skip	1
	.size	S, 1

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
	.int8	52                              # DW_TAG_variable
	.int8	0                               # DW_CHILDREN_no
	.int8	3                               # DW_AT_name
	.int8	14                              # DW_FORM_strp
	.int8	73                              # DW_AT_type
	.int8	19                              # DW_FORM_ref4
	.int8	63                              # DW_AT_external
	.int8	25                              # DW_FORM_flag_present
	.int8	58                              # DW_AT_decl_file
	.int8	11                              # DW_FORM_data1
	.int8	59                              # DW_AT_decl_line
	.int8	11                              # DW_FORM_data1
	.int8	2                               # DW_AT_location
	.int8	24                              # DW_FORM_exprloc
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	3                               # Abbreviation Code
	.int8	19                              # DW_TAG_structure_type
	.int8	1                               # DW_CHILDREN_yes
	.int8	54                              # DW_AT_calling_convention
	.int8	11                              # DW_FORM_data1
	.int8	3                               # DW_AT_name
	.int8	14                              # DW_FORM_strp
	.int8	11                              # DW_AT_byte_size
	.int8	11                              # DW_FORM_data1
	.int8	58                              # DW_AT_decl_file
	.int8	11                              # DW_FORM_data1
	.int8	59                              # DW_AT_decl_line
	.int8	11                              # DW_FORM_data1
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	4                               # Abbreviation Code
	.int8	13                              # DW_TAG_member
	.int8	0                               # DW_CHILDREN_no
	.int8	3                               # DW_AT_name
	.int8	14                              # DW_FORM_strp
	.int8	73                              # DW_AT_type
	.int8	19                              # DW_FORM_ref4
	.int8	58                              # DW_AT_decl_file
	.int8	11                              # DW_FORM_data1
	.int8	59                              # DW_AT_decl_line
	.int8	11                              # DW_FORM_data1
	.int8	56                              # DW_AT_data_member_location
	.int8	11                              # DW_FORM_data1
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	5                               # Abbreviation Code
	.int8	23                              # DW_TAG_union_type
	.int8	0                               # DW_CHILDREN_no
	.int8	54                              # DW_AT_calling_convention
	.int8	11                              # DW_FORM_data1
	.int8	3                               # DW_AT_name
	.int8	14                              # DW_FORM_strp
	.int8	11                              # DW_AT_byte_size
	.int8	11                              # DW_FORM_data1
	.int8	58                              # DW_AT_decl_file
	.int8	11                              # DW_FORM_data1
	.int8	59                              # DW_AT_decl_line
	.int8	11                              # DW_FORM_data1
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	6                               # Abbreviation Code
	.int8	46                              # DW_TAG_subprogram
	.int8	0                               # DW_CHILDREN_no
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
	.int8	7                               # Abbreviation Code
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
	.int8	0                               # EOM(3)
	.section	.debug_info,"",@
.Lcu_begin0:
	.int32	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.int16	4                               # DWARF version number
	.int32	.debug_abbrev0                  # Offset Into Abbrev. Section
	.int8	4                               # Address Size (in bytes)
	.int8	1                               # Abbrev [1] 0xb:0x72 DW_TAG_compile_unit
	.int32	.Linfo_string0                  # DW_AT_producer
	.int16	33                              # DW_AT_language
	.int32	.Linfo_string1                  # DW_AT_name
	.int32	.Lline_table_start0             # DW_AT_stmt_list
	.int32	.Linfo_string2                  # DW_AT_comp_dir
	.int32	.Lfunc_begin0                   # DW_AT_low_pc
	.int32	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.int8	2                               # Abbrev [2] 0x26:0x11 DW_TAG_variable
	.int32	.Linfo_string3                  # DW_AT_name
	.int32	55                              # DW_AT_type
                                        # DW_AT_external
	.int8	1                               # DW_AT_decl_file
	.int8	8                               # DW_AT_decl_line
	.int8	5                               # DW_AT_location
	.int8	3
	.int32	S
	.int8	3                               # Abbrev [3] 0x37:0x1f DW_TAG_structure_type
	.int8	5                               # DW_AT_calling_convention
	.int32	.Linfo_string6                  # DW_AT_name
	.int8	1                               # DW_AT_byte_size
	.int8	1                               # DW_AT_decl_file
	.int8	1                               # DW_AT_decl_line
	.int8	4                               # Abbrev [4] 0x40:0xc DW_TAG_member
	.int32	.Linfo_string4                  # DW_AT_name
	.int32	76                              # DW_AT_type
	.int8	1                               # DW_AT_decl_file
	.int8	5                               # DW_AT_decl_line
	.int8	0                               # DW_AT_data_member_location
	.int8	5                               # Abbrev [5] 0x4c:0x9 DW_TAG_union_type
	.int8	5                               # DW_AT_calling_convention
	.int32	.Linfo_string5                  # DW_AT_name
	.int8	1                               # DW_AT_byte_size
	.int8	1                               # DW_AT_decl_file
	.int8	2                               # DW_AT_decl_line
	.int8	0                               # End Of Children Mark
	.int8	6                               # Abbrev [6] 0x56:0x1f DW_TAG_subprogram
	.int32	.Lfunc_begin0                   # DW_AT_low_pc
	.int32	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.int8	7                               # DW_AT_frame_base
	.int8	237
	.int8	3
	.int32	__stack_pointer
	.int8	159
	.int32	.Linfo_string7                  # DW_AT_linkage_name
	.int32	.Linfo_string8                  # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	9                               # DW_AT_decl_line
	.int32	117                             # DW_AT_type
                                        # DW_AT_external
	.int8	7                               # Abbrev [7] 0x75:0x7 DW_TAG_base_type
	.int32	.Linfo_string9                  # DW_AT_name
	.int8	5                               # DW_AT_encoding
	.int8	4                               # DW_AT_byte_size
	.int8	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"S",@
.Linfo_string0:
	.asciz	"clang version 19.0.0git (/data/projects/llvm-root/llvm-project/clang 2db6703f0c257d293df455e2dff8c1fb695c4100)" # string offset=0
.Linfo_string1:
	.asciz	"pr-46466.cpp"                  # string offset=111
.Linfo_string2:
	.asciz	"/data/projects/scripts/regression-suite/input/general" # string offset=124
.Linfo_string3:
	.asciz	"S"                             # string offset=178
.Linfo_string4:
	.asciz	"U"                             # string offset=180
.Linfo_string5:
	.asciz	"Union"                         # string offset=182
.Linfo_string6:
	.asciz	"Struct"                        # string offset=188
.Linfo_string7:
	.asciz	"_Z4testv"                      # string offset=195
.Linfo_string8:
	.asciz	"test"                          # string offset=204
.Linfo_string9:
	.asciz	"int"                           # string offset=209
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
