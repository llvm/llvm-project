	.text
	.file	"test.cpp"
	.globaltype	__stack_pointer, i32
	.functype	_Z3fooPKijb (i32, i32, i32) -> (i32)
	.section	.text._Z3fooPKijb,"",@
	.hidden	_Z3fooPKijb                     # -- Begin function _Z3fooPKijb
	.globl	_Z3fooPKijb
	.type	_Z3fooPKijb,@function
_Z3fooPKijb:                            # @_Z3fooPKijb
.Lfunc_begin0:
	.file	1 "/data/projects/scripts/regression-suite/input/general" "test.cpp"
	.loc	1 2 0                           # test.cpp:2:0
	.functype	_Z3fooPKijb (i32, i32, i32) -> (i32)
	.local  	i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
# %bb.0:                                # %entry
	global.get	__stack_pointer
	local.set	3
	i32.const	32
	local.set	4
	local.get	3
	local.get	4
	i32.sub 
	local.set	5
	local.get	5
	local.get	0
	i32.store	24
	local.get	5
	local.get	1
	i32.store	20
	local.get	2
	local.set	6
	local.get	5
	local.get	6
	i32.store8	19
.Ltmp0:
	.loc	1 3 7 prologue_end              # test.cpp:3:7
	local.get	5
	i32.load8_u	19
	local.set	7
.Ltmp1:
	.loc	1 3 7 is_stmt 0                 # test.cpp:3:7
	i32.const	1
	local.set	8
	local.get	7
	local.get	8
	i32.and 
	local.set	9
	block   	
	block   	
	local.get	9
	i32.eqz
	br_if   	0                               # 0: down to label1
# %bb.1:                                # %if.then
.Ltmp2:
	.loc	1 5 19 is_stmt 1                # test.cpp:5:19
	i32.const	7
	local.set	10
	local.get	5
	local.get	10
	i32.store	12
	.loc	1 6 5                           # test.cpp:6:5
	i32.const	7
	local.set	11
	local.get	5
	local.get	11
	i32.store	28
	br      	1                               # 1: down to label0
.Ltmp3:
.LBB0_2:                                # %if.end
	.loc	1 0 5 is_stmt 0                 # test.cpp:0:5
	end_block                               # label1:
	.loc	1 8 10 is_stmt 1                # test.cpp:8:10
	local.get	5
	i32.load	20
	local.set	12
	.loc	1 8 3 is_stmt 0                 # test.cpp:8:3
	local.get	5
	local.get	12
	i32.store	28
.LBB0_3:                                # %return
	.loc	1 0 3                           # test.cpp:0:3
	end_block                               # label0:
	.loc	1 9 1 is_stmt 1                 # test.cpp:9:1
	local.get	5
	i32.load	28
	local.set	13
	local.get	13
	return
	end_function
.Ltmp4:
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
	.int8	3                               # Abbreviation Code
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
	.int8	4                               # Abbreviation Code
	.int8	11                              # DW_TAG_lexical_block
	.int8	1                               # DW_CHILDREN_yes
	.int8	17                              # DW_AT_low_pc
	.int8	1                               # DW_FORM_addr
	.int8	18                              # DW_AT_high_pc
	.int8	6                               # DW_FORM_data4
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	5                               # Abbreviation Code
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
	.int8	6                               # Abbreviation Code
	.int8	22                              # DW_TAG_typedef
	.int8	0                               # DW_CHILDREN_no
	.int8	73                              # DW_AT_type
	.int8	19                              # DW_FORM_ref4
	.int8	3                               # DW_AT_name
	.int8	14                              # DW_FORM_strp
	.int8	58                              # DW_AT_decl_file
	.int8	11                              # DW_FORM_data1
	.int8	59                              # DW_AT_decl_line
	.int8	11                              # DW_FORM_data1
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
	.int8	8                               # Abbreviation Code
	.int8	15                              # DW_TAG_pointer_type
	.int8	0                               # DW_CHILDREN_no
	.int8	73                              # DW_AT_type
	.int8	19                              # DW_FORM_ref4
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	9                               # Abbreviation Code
	.int8	38                              # DW_TAG_const_type
	.int8	0                               # DW_CHILDREN_no
	.int8	73                              # DW_AT_type
	.int8	19                              # DW_FORM_ref4
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
	.int8	1                               # Abbrev [1] 0xb:0xb5 DW_TAG_compile_unit
	.int32	.Linfo_string0                  # DW_AT_producer
	.int16	33                              # DW_AT_language
	.int32	.Linfo_string1                  # DW_AT_name
	.int32	.Lline_table_start0             # DW_AT_stmt_list
	.int32	.Linfo_string2                  # DW_AT_comp_dir
	.int32	.Lfunc_begin0                   # DW_AT_low_pc
	.int32	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.int8	2                               # Abbrev [2] 0x26:0x6a DW_TAG_subprogram
	.int32	.Lfunc_begin0                   # DW_AT_low_pc
	.int32	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.int8	4                               # DW_AT_frame_base
	.int8	237
	.int8	0
	.int8	5
	.int8	159
	.int32	.Linfo_string3                  # DW_AT_linkage_name
	.int32	.Linfo_string4                  # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	2                               # DW_AT_decl_line
	.int32	144                             # DW_AT_type
                                        # DW_AT_external
	.int8	3                               # Abbrev [3] 0x42:0xe DW_TAG_formal_parameter
	.int8	2                               # DW_AT_location
	.int8	145
	.int8	24
	.int32	.Linfo_string6                  # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	2                               # DW_AT_decl_line
	.int32	151                             # DW_AT_type
	.int8	3                               # Abbrev [3] 0x50:0xe DW_TAG_formal_parameter
	.int8	2                               # DW_AT_location
	.int8	145
	.int8	20
	.int32	.Linfo_string8                  # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	2                               # DW_AT_decl_line
	.int32	172                             # DW_AT_type
	.int8	3                               # Abbrev [3] 0x5e:0xe DW_TAG_formal_parameter
	.int8	2                               # DW_AT_location
	.int8	145
	.int8	19
	.int32	.Linfo_string10                 # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	2                               # DW_AT_decl_line
	.int32	179                             # DW_AT_type
	.int8	4                               # Abbrev [4] 0x6c:0x18 DW_TAG_lexical_block
	.int32	.Ltmp2                          # DW_AT_low_pc
	.int32	.Ltmp3-.Ltmp2                   # DW_AT_high_pc
	.int8	5                               # Abbrev [5] 0x75:0xe DW_TAG_variable
	.int8	2                               # DW_AT_location
	.int8	145
	.int8	12
	.int32	.Linfo_string12                 # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	5                               # DW_AT_decl_line
	.int32	186                             # DW_AT_type
	.int8	0                               # End Of Children Mark
	.int8	6                               # Abbrev [6] 0x84:0xb DW_TAG_typedef
	.int32	144                             # DW_AT_type
	.int32	.Linfo_string13                 # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	4                               # DW_AT_decl_line
	.int8	0                               # End Of Children Mark
	.int8	7                               # Abbrev [7] 0x90:0x7 DW_TAG_base_type
	.int32	.Linfo_string5                  # DW_AT_name
	.int8	5                               # DW_AT_encoding
	.int8	4                               # DW_AT_byte_size
	.int8	6                               # Abbrev [6] 0x97:0xb DW_TAG_typedef
	.int32	162                             # DW_AT_type
	.int32	.Linfo_string7                  # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	1                               # DW_AT_decl_line
	.int8	8                               # Abbrev [8] 0xa2:0x5 DW_TAG_pointer_type
	.int32	167                             # DW_AT_type
	.int8	9                               # Abbrev [9] 0xa7:0x5 DW_TAG_const_type
	.int32	144                             # DW_AT_type
	.int8	7                               # Abbrev [7] 0xac:0x7 DW_TAG_base_type
	.int32	.Linfo_string9                  # DW_AT_name
	.int8	7                               # DW_AT_encoding
	.int8	4                               # DW_AT_byte_size
	.int8	7                               # Abbrev [7] 0xb3:0x7 DW_TAG_base_type
	.int32	.Linfo_string11                 # DW_AT_name
	.int8	2                               # DW_AT_encoding
	.int8	1                               # DW_AT_byte_size
	.int8	9                               # Abbrev [9] 0xba:0x5 DW_TAG_const_type
	.int32	132                             # DW_AT_type
	.int8	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"S",@
.Linfo_string0:
	.asciz	"clang version 19.0.0git (/data/projects/llvm-root/llvm-project/clang 2db6703f0c257d293df455e2dff8c1fb695c4100)" # string offset=0
.Linfo_string1:
	.asciz	"test.cpp"                      # string offset=111
.Linfo_string2:
	.asciz	"/data/projects/scripts/regression-suite/input/general" # string offset=120
.Linfo_string3:
	.asciz	"_Z3fooPKijb"                   # string offset=174
.Linfo_string4:
	.asciz	"foo"                           # string offset=186
.Linfo_string5:
	.asciz	"int"                           # string offset=190
.Linfo_string6:
	.asciz	"ParamPtr"                      # string offset=194
.Linfo_string7:
	.asciz	"INTPTR"                        # string offset=203
.Linfo_string8:
	.asciz	"ParamUnsigned"                 # string offset=210
.Linfo_string9:
	.asciz	"unsigned int"                  # string offset=224
.Linfo_string10:
	.asciz	"ParamBool"                     # string offset=237
.Linfo_string11:
	.asciz	"bool"                          # string offset=247
.Linfo_string12:
	.asciz	"CONSTANT"                      # string offset=252
.Linfo_string13:
	.asciz	"INTEGER"                       # string offset=261
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
