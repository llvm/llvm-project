	.text
	.file	"hello-world.cpp"
	.file	1 "/data/projects/scripts/regression-suite/input/general" "hello-world.cpp"
	.globaltype	__stack_pointer, i32
	.functype	__original_main () -> (i32)
	.functype	_Z6printfPKcz (i32, i32) -> (i32)
	.functype	main (i32, i32) -> (i32)
	.section	.text.__original_main,"",@
	.hidden	__original_main                 # -- Begin function __original_main
	.globl	__original_main
	.type	__original_main,@function
__original_main:                        # @__original_main
.Lfunc_begin0:
	.loc	1 4 0                           # hello-world.cpp:4:0
	.functype	__original_main () -> (i32)
	.local  	i32, i32, i32, i32, i32, i32, i32, i32, i32
# %bb.0:                                # %entry
	global.get	__stack_pointer
	local.set	0
	i32.const	16
	local.set	1
	local.get	0
	local.get	1
	i32.sub 
	local.set	2
	local.get	2
	global.set	__stack_pointer
	i32.const	0
	local.set	3
	local.get	2
	local.get	3
	i32.store	12
.Ltmp0:
	.loc	1 5 3 prologue_end              # hello-world.cpp:5:3
	i32.const	.L.str
	local.set	4
	i32.const	0
	local.set	5
	local.get	4
	local.get	5
	call	_Z6printfPKcz
	drop
	.loc	1 6 3                           # hello-world.cpp:6:3
	i32.const	0
	local.set	6
	i32.const	16
	local.set	7
	local.get	2
	local.get	7
	i32.add 
	local.set	8
	local.get	8
	global.set	__stack_pointer
	local.get	6
	return
	end_function
.Ltmp1:
.Lfunc_end0:
                                        # -- End function
	.section	.text.main,"",@
	.hidden	main                            # -- Begin function main
	.globl	main
	.type	main,@function
main:                                   # @main
.Lfunc_begin1:
	.functype	main (i32, i32) -> (i32)
	.local  	i32
# %bb.0:                                # %body
	call	__original_main
	local.set	2
	local.get	2
	return
	end_function
.Lfunc_end1:
                                        # -- End function
	.type	.L.str,@object                  # @.str
	.section	.rodata..L.str,"S",@
.L.str:
	.asciz	"Hello, World\n"
	.size	.L.str, 14

	.globl	__main_void
	.type	__main_void,@function
	.hidden	__main_void
.set __main_void, __original_main
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
	.int8	73                              # DW_AT_type
	.int8	19                              # DW_FORM_ref4
	.int8	58                              # DW_AT_decl_file
	.int8	11                              # DW_FORM_data1
	.int8	59                              # DW_AT_decl_line
	.int8	11                              # DW_FORM_data1
	.int8	2                               # DW_AT_location
	.int8	24                              # DW_FORM_exprloc
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	3                               # Abbreviation Code
	.int8	1                               # DW_TAG_array_type
	.int8	1                               # DW_CHILDREN_yes
	.int8	73                              # DW_AT_type
	.int8	19                              # DW_FORM_ref4
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	4                               # Abbreviation Code
	.int8	33                              # DW_TAG_subrange_type
	.int8	0                               # DW_CHILDREN_no
	.int8	73                              # DW_AT_type
	.int8	19                              # DW_FORM_ref4
	.int8	55                              # DW_AT_count
	.int8	11                              # DW_FORM_data1
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	5                               # Abbreviation Code
	.int8	38                              # DW_TAG_const_type
	.int8	0                               # DW_CHILDREN_no
	.int8	73                              # DW_AT_type
	.int8	19                              # DW_FORM_ref4
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
	.int8	36                              # DW_TAG_base_type
	.int8	0                               # DW_CHILDREN_no
	.int8	3                               # DW_AT_name
	.int8	14                              # DW_FORM_strp
	.int8	11                              # DW_AT_byte_size
	.int8	11                              # DW_FORM_data1
	.int8	62                              # DW_AT_encoding
	.int8	11                              # DW_FORM_data1
	.int8	0                               # EOM(1)
	.int8	0                               # EOM(2)
	.int8	8                               # Abbreviation Code
	.int8	46                              # DW_TAG_subprogram
	.int8	0                               # DW_CHILDREN_no
	.int8	17                              # DW_AT_low_pc
	.int8	1                               # DW_FORM_addr
	.int8	18                              # DW_AT_high_pc
	.int8	6                               # DW_FORM_data4
	.int8	64                              # DW_AT_frame_base
	.int8	24                              # DW_FORM_exprloc
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
	.int8	0                               # EOM(3)
	.section	.debug_info,"",@
.Lcu_begin0:
	.int32	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.int16	4                               # DWARF version number
	.int32	.debug_abbrev0                  # Offset Into Abbrev. Section
	.int8	4                               # Address Size (in bytes)
	.int8	1                               # Abbrev [1] 0xb:0x67 DW_TAG_compile_unit
	.int32	.Linfo_string0                  # DW_AT_producer
	.int16	33                              # DW_AT_language
	.int32	.Linfo_string1                  # DW_AT_name
	.int32	.Lline_table_start0             # DW_AT_stmt_list
	.int32	.Linfo_string2                  # DW_AT_comp_dir
	.int32	.Lfunc_begin0                   # DW_AT_low_pc
	.int32	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.int8	2                               # Abbrev [2] 0x26:0xd DW_TAG_variable
	.int32	51                              # DW_AT_type
	.int8	1                               # DW_AT_decl_file
	.int8	5                               # DW_AT_decl_line
	.int8	5                               # DW_AT_location
	.int8	3
	.int32	.L.str
	.int8	3                               # Abbrev [3] 0x33:0xc DW_TAG_array_type
	.int32	63                              # DW_AT_type
	.int8	4                               # Abbrev [4] 0x38:0x6 DW_TAG_subrange_type
	.int32	75                              # DW_AT_type
	.int8	14                              # DW_AT_count
	.int8	0                               # End Of Children Mark
	.int8	5                               # Abbrev [5] 0x3f:0x5 DW_TAG_const_type
	.int32	68                              # DW_AT_type
	.int8	6                               # Abbrev [6] 0x44:0x7 DW_TAG_base_type
	.int32	.Linfo_string3                  # DW_AT_name
	.int8	6                               # DW_AT_encoding
	.int8	1                               # DW_AT_byte_size
	.int8	7                               # Abbrev [7] 0x4b:0x7 DW_TAG_base_type
	.int32	.Linfo_string4                  # DW_AT_name
	.int8	8                               # DW_AT_byte_size
	.int8	7                               # DW_AT_encoding
	.int8	8                               # Abbrev [8] 0x52:0x18 DW_TAG_subprogram
	.int32	.Lfunc_begin0                   # DW_AT_low_pc
	.int32	.Lfunc_end0-.Lfunc_begin0       # DW_AT_high_pc
	.int8	4                               # DW_AT_frame_base
	.int8	237
	.int8	0
	.int8	2
	.int8	159
	.int32	.Linfo_string5                  # DW_AT_name
	.int8	1                               # DW_AT_decl_file
	.int8	3                               # DW_AT_decl_line
	.int32	106                             # DW_AT_type
                                        # DW_AT_external
	.int8	6                               # Abbrev [6] 0x6a:0x7 DW_TAG_base_type
	.int32	.Linfo_string6                  # DW_AT_name
	.int8	5                               # DW_AT_encoding
	.int8	4                               # DW_AT_byte_size
	.int8	0                               # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str,"S",@
.Linfo_string0:
	.asciz	"clang version 19.0.0git (/data/projects/llvm-root/llvm-project/clang 2db6703f0c257d293df455e2dff8c1fb695c4100)" # string offset=0
.Linfo_string1:
	.asciz	"hello-world.cpp"               # string offset=111
.Linfo_string2:
	.asciz	"/data/projects/scripts/regression-suite/input/general" # string offset=127
.Linfo_string3:
	.asciz	"char"                          # string offset=181
.Linfo_string4:
	.asciz	"__ARRAY_SIZE_TYPE__"           # string offset=186
.Linfo_string5:
	.asciz	"main"                          # string offset=206
.Linfo_string6:
	.asciz	"int"                           # string offset=211
	.ident	"clang version 19.0.0git (/data/projects/llvm-root/llvm-project/clang 2db6703f0c257d293df455e2dff8c1fb695c4100)"
	.no_dead_strip	__indirect_function_table
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
