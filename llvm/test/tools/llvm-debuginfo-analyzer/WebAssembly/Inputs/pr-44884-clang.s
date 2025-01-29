	.text
	.file	"pr-44884.cpp"
	.globaltype	__stack_pointer, i32
	.functype	_Z3barf (f32) -> (i32)
	.functype	_Z3fooc (i32) -> (i32)
	.section	.text._Z3barf,"",@
	.hidden	_Z3barf                         # -- Begin function _Z3barf
	.globl	_Z3barf
	.type	_Z3barf,@function
_Z3barf:                                # @_Z3barf
.Lfunc_begin0:
	.file	1 "/data/projects/scripts/regression-suite/input/general" "pr-44884.cpp"
	.loc	1 1 0                           # pr-44884.cpp:1:0
	.functype	_Z3barf (f32) -> (i32)
	.local  	i32, i32, i32, f32, f32, f32, i32, i32, i32, i32, i32, i32
# %bb.0:                                # %entry
	global.get	__stack_pointer
	local.set	1
	i32.const	16
	local.set	2
	local.get	1
	local.get	2
	i32.sub 
	local.set	3
	local.get	3
	local.get	0
	f32.store	12
.Ltmp0:
	.loc	1 1 36 prologue_end             # pr-44884.cpp:1:36
	local.get	3
	f32.load	12
	local.set	4
	local.get	4
	f32.abs 
	local.set	5
	f32.const	0x1p31
	local.set	6
	local.get	5
	local.get	6
	f32.lt  
	local.set	7
	local.get	7
	i32.eqz
	local.set	8
	block   	
	block   	
	local.get	8
	br_if   	0                               # 0: down to label1
# %bb.1:                                # %entry
	local.get	4
	i32.trunc_f32_s
	local.set	9
	local.get	9
	local.set	10
	br      	1                               # 1: down to label0
.LBB0_2:                                # %entry
	.loc	1 0 36 is_stmt 0                # pr-44884.cpp:0:36
	end_block                               # label1:
	.loc	1 1 36                          # pr-44884.cpp:1:36
	i32.const	-2147483648
	local.set	11
	local.get	11
	local.set	10
.LBB0_3:                                # %entry
	.loc	1 0 36                          # pr-44884.cpp:0:36
	end_block                               # label0:
	.loc	1 1 36                          # pr-44884.cpp:1:36
	local.get	10
	local.set	12
	.loc	1 1 24                          # pr-44884.cpp:1:24
	local.get	12
	return
	end_function
.Ltmp1:
.Lfunc_end0:
                                        # -- End function
	.section	.text._Z3fooc,"",@
	.hidden	_Z3fooc                         # -- Begin function _Z3fooc
	.globl	_Z3fooc
	.type	_Z3fooc,@function
_Z3fooc:                                # @_Z3fooc
.Lfunc_begin1:
	.loc	1 3 0 is_stmt 1                 # pr-44884.cpp:3:0
	.functype	_Z3fooc (i32) -> (i32)
	.local  	i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, f32, f32, i32, i32, i32, i32, i32, i32, i32, i32, i32
# %bb.0:                                # %entry
	global.get	__stack_pointer
	local.set	1
	i32.const	16
	local.set	2
	local.get	1
	local.get	2
	i32.sub 
	local.set	3
	local.get	3
	global.set	__stack_pointer
	local.get	3
	local.get	0
	i32.store8	15
.Ltmp2:
	.loc	1 5 15 prologue_end             # pr-44884.cpp:5:15
	local.get	3
	i32.load8_u	15
	local.set	4
	i32.const	24
	local.set	5
	local.get	4
	local.get	5
	i32.shl 
	local.set	6
	local.get	6
	local.get	5
	i32.shr_s
	local.set	7
	.loc	1 5 7 is_stmt 0                 # pr-44884.cpp:5:7
	local.get	3
	local.get	7
	i32.store	8
.Ltmp3:
	.loc	1 9 21 is_stmt 1                # pr-44884.cpp:9:21
	local.get	3
	i32.load	8
	local.set	8
	.loc	1 9 29 is_stmt 0                # pr-44884.cpp:9:29
	local.get	3
	i32.load8_u	15
	local.set	9
	i32.const	24
	local.set	10
	local.get	9
	local.get	10
	i32.shl 
	local.set	11
	local.get	11
	local.get	10
	i32.shr_s
	local.set	12
	.loc	1 9 27                          # pr-44884.cpp:9:27
	local.get	8
	local.get	12
	i32.add 
	local.set	13
	.loc	1 9 21                          # pr-44884.cpp:9:21
	local.get	13
	f32.convert_i32_s
	local.set	14
	.loc	1 9 13                          # pr-44884.cpp:9:13
	local.get	3
	local.get	14
	f32.store	4
	.loc	1 10 19 is_stmt 1               # pr-44884.cpp:10:19
	local.get	3
	f32.load	4
	local.set	15
	.loc	1 10 15 is_stmt 0               # pr-44884.cpp:10:15
	local.get	15
	call	_Z3barf
	local.set	16
	.loc	1 10 13                         # pr-44884.cpp:10:13
	local.get	3
	local.get	16
	i32.store	8
.Ltmp4:
	.loc	1 13 10 is_stmt 1               # pr-44884.cpp:13:10
	local.get	3
	i32.load	8
	local.set	17
	.loc	1 13 18 is_stmt 0               # pr-44884.cpp:13:18
	local.get	3
	i32.load8_u	15
	local.set	18
	i32.const	24
	local.set	19
	local.get	18
	local.get	19
	i32.shl 
	local.set	20
	local.get	20
	local.get	19
	i32.shr_s
	local.set	21
	.loc	1 13 16                         # pr-44884.cpp:13:16
	local.get	17
	local.get	21
	i32.add 
	local.set	22
	.loc	1 13 3                          # pr-44884.cpp:13:3
	i32.const	16
	local.set	23
	local.get	3
	local.get	23
	i32.add 
	local.set	24
	local.get	24
	global.set	__stack_pointer
	local.get	22
	return
	end_function
.Ltmp5:
.Lfunc_end1:
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
	.int8	85                              # DW_AT_ranges
	.int8	23                              # DW_FORM_sec_offset
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	2                               # Abbreviation Code
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
	.int8	3                               # Abbreviation Code
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
	.int8	4                               # Abbreviation Code
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
	.int8	11                              # DW_TAG_lexical_block
	.int8	1                               # DW_CHILDREN_yes
	.int8	17                              # DW_AT_low_pc
	.int8	1                               # DW_FORM_addr
	.int8	18                              # DW_AT_high_pc
	.int8	6                               # DW_FORM_data4
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	7                               # Abbreviation Code
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
	.int8	0                               # EOM(3)
	.section	.debug_info,"",@
.Lcu_begin0:
	.int32	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.int16	4                               # DWARF version number
	.int32	.debug_abbrev0                  # Offset Into Abbrev. Section
	.int8	4                               # Address Size (in bytes)
	.int8	1                               # Abbrev [1] 0xb:0xca DW_TAG_compile_unit
	.int32	.Linfo_string0                  # DW_AT_producer
	.int16	33                              # DW_AT_language
	.int32	.Linfo_string1                  # DW_AT_name
	.int32	.Lline_table_start0             # DW_AT_stmt_list
	.int32	.Linfo_string2                  # DW_AT_comp_dir
	.int32	0                               # DW_AT_low_pc
	.int32	.Ldebug_ranges0                 # DW_AT_ranges
	.int8	2                               # Abbrev [2] 0x26:0x7 DW_TAG_base_type
	.int32	.Linfo_string3                  # DW_AT_name
	.int8	5                               # DW_AT_encoding
	.int8	4                               # DW_AT_byte_size
	.int8	3                               # Abbrev [3] 0x2d:0x2b DW_TAG_subprogram
	.int32	.Lfunc_begin0                   # DW_AT_low_pc
	.int32	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.int8	4                               # DW_AT_frame_base
	.int8	237
	.int8	0
	.int8	3
	.int8	159
	.int32	.Linfo_string4                  # DW_AT_linkage_name
	.int32	.Linfo_string5                  # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	1                               # DW_AT_decl_line
	.int32	38                              # DW_AT_type
                                        # DW_AT_external
	.int8	4                               # Abbrev [4] 0x49:0xe DW_TAG_formal_parameter
	.int8	2                               # DW_AT_location
	.int8	145
	.int8	12
	.int32	.Linfo_string9                  # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	1                               # DW_AT_decl_line
	.int32	198                             # DW_AT_type
	.int8	0                               # End Of Children Mark
	.int8	3                               # Abbrev [3] 0x58:0x67 DW_TAG_subprogram
	.int32	.Lfunc_begin1                   # DW_AT_low_pc
	.int32	.Lfunc_end1-.Lfunc_begin1       # DW_AT_high_pc
	.int8	4                               # DW_AT_frame_base
	.int8	237
	.int8	0
	.int8	3
	.int8	159
	.int32	.Linfo_string6                  # DW_AT_linkage_name
	.int32	.Linfo_string7                  # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	3                               # DW_AT_decl_line
	.int32	191                             # DW_AT_type
                                        # DW_AT_external
	.int8	4                               # Abbrev [4] 0x74:0xe DW_TAG_formal_parameter
	.int8	2                               # DW_AT_location
	.int8	145
	.int8	15
	.int32	.Linfo_string11                 # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	3                               # DW_AT_decl_line
	.int32	205                             # DW_AT_type
	.int8	5                               # Abbrev [5] 0x82:0xe DW_TAG_variable
	.int8	2                               # DW_AT_location
	.int8	145
	.int8	8
	.int32	.Linfo_string13                 # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	5                               # DW_AT_decl_line
	.int32	168                             # DW_AT_type
	.int8	6                               # Abbrev [6] 0x90:0x18 DW_TAG_lexical_block
	.int32	.Ltmp3                          # DW_AT_low_pc
	.int32	.Ltmp4-.Ltmp3                   # DW_AT_high_pc
	.int8	5                               # Abbrev [5] 0x99:0xe DW_TAG_variable
	.int8	2                               # DW_AT_location
	.int8	145
	.int8	4
	.int32	.Linfo_string15                 # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	9                               # DW_AT_decl_line
	.int32	179                             # DW_AT_type
	.int8	0                               # End Of Children Mark
	.int8	7                               # Abbrev [7] 0xa8:0xb DW_TAG_typedef
	.int32	38                              # DW_AT_type
	.int32	.Linfo_string14                 # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	4                               # DW_AT_decl_line
	.int8	7                               # Abbrev [7] 0xb3:0xb DW_TAG_typedef
	.int32	198                             # DW_AT_type
	.int32	.Linfo_string16                 # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	7                               # DW_AT_decl_line
	.int8	0                               # End Of Children Mark
	.int8	2                               # Abbrev [2] 0xbf:0x7 DW_TAG_base_type
	.int32	.Linfo_string8                  # DW_AT_name
	.int8	7                               # DW_AT_encoding
	.int8	4                               # DW_AT_byte_size
	.int8	2                               # Abbrev [2] 0xc6:0x7 DW_TAG_base_type
	.int32	.Linfo_string10                 # DW_AT_name
	.int8	4                               # DW_AT_encoding
	.int8	4                               # DW_AT_byte_size
	.int8	2                               # Abbrev [2] 0xcd:0x7 DW_TAG_base_type
	.int32	.Linfo_string12                 # DW_AT_name
	.int8	6                               # DW_AT_encoding
	.int8	1                               # DW_AT_byte_size
	.int8	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@
.Ldebug_ranges0:
	.int32	.Lfunc_begin0
	.int32	.Lfunc_end0
	.int32	.Lfunc_begin1
	.int32	.Lfunc_end1
	.int32	0
	.int32	0
	.section	.debug_str,"S",@
.Linfo_string0:
	.asciz	"clang version 19.0.0git (/data/projects/llvm-root/llvm-project/clang 2db6703f0c257d293df455e2dff8c1fb695c4100)" # string offset=0
.Linfo_string1:
	.asciz	"pr-44884.cpp"                  # string offset=111
.Linfo_string2:
	.asciz	"/data/projects/scripts/regression-suite/input/general" # string offset=124
.Linfo_string3:
	.asciz	"int"                           # string offset=178
.Linfo_string4:
	.asciz	"_Z3barf"                       # string offset=182
.Linfo_string5:
	.asciz	"bar"                           # string offset=190
.Linfo_string6:
	.asciz	"_Z3fooc"                       # string offset=194
.Linfo_string7:
	.asciz	"foo"                           # string offset=202
.Linfo_string8:
	.asciz	"unsigned int"                  # string offset=206
.Linfo_string9:
	.asciz	"Input"                         # string offset=219
.Linfo_string10:
	.asciz	"float"                         # string offset=225
.Linfo_string11:
	.asciz	"Param"                         # string offset=231
.Linfo_string12:
	.asciz	"char"                          # string offset=237
.Linfo_string13:
	.asciz	"Value"                         # string offset=242
.Linfo_string14:
	.asciz	"INT"                           # string offset=248
.Linfo_string15:
	.asciz	"Added"                         # string offset=252
.Linfo_string16:
	.asciz	"FLOAT"                         # string offset=258
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
