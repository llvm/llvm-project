# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -dwarf-version=5 -o %t.o
# RUN: llvm-dwarfdump %t.o | FileCheck %s
# RUN: llvm-dwarfdump -verify %t.o

# CHECK: DW_AT_location  (DW_OP_constu 0x7b, DW_OP_convert 0x0, DW_OP_stack_value)

	.text
	.file	"const.ll"
	.globl	foo                     # -- Begin function foo
	.p2align	4, 0x90
	.type	foo,@function
foo:                                    # @foo
.Lfoo$local:
.Lfunc_begin0:
	.file	0 "/const.c"
	.loc	0 1 0                   # const.c:1:0
	.cfi_startproc
# %bb.0:                                # %entry
	.loc	0 3 1 prologue_end      # const.c:3:1
	retq
.Ltmp0:
.Lfunc_end0:
	.size	foo, .Lfunc_end0-foo
	.cfi_endproc
                                        # -- End function
	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	37                      # DW_AT_producer
	.byte	37                      # DW_FORM_strx1
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	3                       # DW_AT_name
	.byte	37                      # DW_FORM_strx1
	.byte	114                     # DW_AT_str_offsets_base
	.byte	23                      # DW_FORM_sec_offset
	.byte	16                      # DW_AT_stmt_list
	.byte	23                      # DW_FORM_sec_offset
	.byte	27                      # DW_AT_comp_dir
	.byte	37                      # DW_FORM_strx1
	.byte	17                      # DW_AT_low_pc
	.byte	27                      # DW_FORM_addrx
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	115                     # DW_AT_addr_base
	.byte	23                      # DW_FORM_sec_offset
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	2                       # Abbreviation Code
	.byte	46                      # DW_TAG_subprogram
	.byte	1                       # DW_CHILDREN_yes
	.byte	17                      # DW_AT_low_pc
	.byte	27                      # DW_FORM_addrx
	.byte	18                      # DW_AT_high_pc
	.byte	6                       # DW_FORM_data4
	.byte	64                      # DW_AT_frame_base
	.byte	24                      # DW_FORM_exprloc
	.byte	122                     # DW_AT_call_all_calls
	.byte	25                      # DW_FORM_flag_present
	.byte	3                       # DW_AT_name
	.byte	37                      # DW_FORM_strx1
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	63                      # DW_AT_external
	.byte	25                      # DW_FORM_flag_present
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	3                       # Abbreviation Code
	.byte	52                      # DW_TAG_variable
	.byte	0                       # DW_CHILDREN_no
	.byte	2                       # DW_AT_location
	.byte	24                      # DW_FORM_exprloc
	.byte	3                       # DW_AT_name
	.byte	37                      # DW_FORM_strx1
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	4                       # Abbreviation Code
	.byte	36                      # DW_TAG_base_type
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	37                      # DW_FORM_strx1
	.byte	62                      # DW_AT_encoding
	.byte	11                      # DW_FORM_data1
	.byte	11                      # DW_AT_byte_size
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	5                       # DWARF version number
	.byte	1                       # DWARF Unit Type
	.byte	8                       # Address Size (in bytes)
	.long	.debug_abbrev           # Offset Into Abbrev. Section
	.byte	1                       # Abbrev [1] 0xc:0x36 DW_TAG_compile_unit
	.byte	0                       # DW_AT_producer
	.short	12                      # DW_AT_language
	.byte	1                       # DW_AT_name
	.long	.Lstr_offsets_base0     # DW_AT_str_offsets_base
	.long	.Lline_table_start0     # DW_AT_stmt_list
	.byte	2                       # DW_AT_comp_dir
	.byte	0                       # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
	.long	.Laddr_table_base0      # DW_AT_addr_base
	.byte	2                       # Abbrev [2] 0x23:0x1a DW_TAG_subprogram
	.byte	0                       # DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0 # DW_AT_high_pc
	.byte	1                       # DW_AT_frame_base
	.byte	87
                                        # DW_AT_call_all_calls
	.byte	3                       # DW_AT_name
	.byte	0                       # DW_AT_decl_file
	.byte	1                       # DW_AT_decl_line
                                        # DW_AT_external
	.byte	3                       # Abbrev [3] 0x2e:0xe DW_TAG_variable
	.byte	5                       # DW_AT_location
	.byte	16
	.byte	123
	.byte	168
	.byte	0
	.byte	159
	.byte	4                       # DW_AT_name
	.byte	0                       # DW_AT_decl_file
	.byte	2                       # DW_AT_decl_line
	.long	61                      # DW_AT_type
	.byte	0                       # End Of Children Mark
	.byte	4                       # Abbrev [4] 0x3d:0x4 DW_TAG_base_type
	.byte	5                       # DW_AT_name
	.byte	5                       # DW_AT_encoding
	.byte	4                       # DW_AT_byte_size
	.byte	0                       # End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_str_offsets,"",@progbits
	.long	28
	.short	5
	.short	0
.Lstr_offsets_base0:
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 11.0.0"  # string offset=0
.Linfo_string1:
	.asciz	"const.c"               # string offset=21
.Linfo_string2:
	.asciz	"/"                     # string offset=29
.Linfo_string3:
	.asciz	"foo"                   # string offset=31
.Linfo_string4:
	.asciz	"local"                 # string offset=35
.Linfo_string5:
	.asciz	"int"                   # string offset=41
	.section	.debug_str_offsets,"",@progbits
	.long	.Linfo_string0
	.long	.Linfo_string1
	.long	.Linfo_string2
	.long	.Linfo_string3
	.long	.Linfo_string4
	.long	.Linfo_string5
	.section	.debug_addr,"",@progbits
	.long	.Ldebug_addr_end0-.Ldebug_addr_start0 # Length of contribution
.Ldebug_addr_start0:
	.short	5                       # DWARF version number
	.byte	8                       # Address size
	.byte	0                       # Segment selector size
.Laddr_table_base0:
	.quad	.Lfunc_begin0
.Ldebug_addr_end0:
	.ident	"clang version 11.0.0"
	.section	".note.GNU-stack","",@progbits
	.section	.debug_line,"",@progbits
.Lline_table_start0:
