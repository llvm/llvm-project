# RUN: llvm-mc < %s -filetype obj -triple x86_64 -o - \
# RUN:   | llvm-dwarfdump - | FileCheck %s

# Generated from:
#
#   struct t1 { };
#   t1 v1;
#
# $ clang++ -S -g -fdebug-types-section -gsplit-dwarf -o test.5.split.s -gdwarf-5 -g

# CHECK: DW_TAG_variable
# CHECK:   DW_AT_type ({{.*}} "t1")

	.text
	.file	"test.cpp"
	.section	.debug_types.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	4                               # DWARF version number
	.long	0                               # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.quad	-4149699470930386446            # Type Signature
	.long	30                              # Type DIE Offset
	.byte	1                               # Abbrev [1] 0x17:0xe DW_TAG_type_unit
	.short	33                              # DW_AT_language
	.long	0                               # DW_AT_stmt_list
	.byte	2                               # Abbrev [2] 0x1e:0x6 DW_TAG_structure_type
	.byte	5                               # DW_AT_calling_convention
	.byte	1                               # DW_AT_name
	.byte	1                               # DW_AT_byte_size
	.byte	1                               # DW_AT_decl_file
	.byte	1                               # DW_AT_decl_line
	.byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end0:
	.file	1 "/usr/local/google/home/blaikie/dev/scratch" "test.cpp"
	.type	v1,@object                      # @v1
	.bss
	.globl	v1
v1:
	.zero	1
	.size	v1, 1

	.section	.debug_abbrev,"",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	27                              # DW_AT_comp_dir
	.byte	14                              # DW_FORM_strp
	.ascii	"\264B"                         # DW_AT_GNU_pubnames
	.byte	25                              # DW_FORM_flag_present
	.ascii	"\260B"                         # DW_AT_GNU_dwo_name
	.byte	14                              # DW_FORM_strp
	.ascii	"\261B"                         # DW_AT_GNU_dwo_id
	.byte	7                               # DW_FORM_data8
	.ascii	"\263B"                         # DW_AT_GNU_addr_base
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
	.short	4                               # DWARF version number
	.long	.debug_abbrev                   # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [1] 0xb:0x19 DW_TAG_compile_unit
	.long	.Lline_table_start0             # DW_AT_stmt_list
	.long	.Lskel_string0                  # DW_AT_comp_dir
                                        # DW_AT_GNU_pubnames
	.long	.Lskel_string1                  # DW_AT_GNU_dwo_name
	.quad	5272896739482311860             # DW_AT_GNU_dwo_id
	.long	.Laddr_table_base0              # DW_AT_GNU_addr_base
.Ldebug_info_end0:
	.section	.debug_str,"MS",@progbits,1
.Lskel_string0:
	.asciz	"/usr/local/google/home/blaikie/dev/scratch" # string offset=0
.Lskel_string1:
	.asciz	"test.dwo"                      # string offset=43
	.section	.debug_str.dwo,"eMS",@progbits,1
.Linfo_string0:
	.asciz	"v1"                            # string offset=0
.Linfo_string1:
	.asciz	"t1"                            # string offset=3
.Linfo_string2:
	.asciz	"clang version 14.0.0 (git@github.com:llvm/llvm-project.git c9e46219f38da5c3fbfe41012173dc893516826e)" # string offset=6
.Linfo_string3:
	.asciz	"test.cpp"                      # string offset=107
.Linfo_string4:
	.asciz	"test.dwo"                      # string offset=116
	.section	.debug_str_offsets.dwo,"e",@progbits
	.long	0
	.long	3
	.long	6
	.long	107
	.long	116
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end1-.Ldebug_info_dwo_start1 # Length of Unit
.Ldebug_info_dwo_start1:
	.short	4                               # DWARF version number
	.long	0                               # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	3                               # Abbrev [3] 0xb:0x23 DW_TAG_compile_unit
	.byte	2                               # DW_AT_producer
	.short	33                              # DW_AT_language
	.byte	3                               # DW_AT_name
	.byte	4                               # DW_AT_GNU_dwo_name
	.quad	5272896739482311860             # DW_AT_GNU_dwo_id
	.byte	4                               # Abbrev [4] 0x19:0xb DW_TAG_variable
	.byte	0                               # DW_AT_name
	.long	36                              # DW_AT_type
                                        # DW_AT_external
	.byte	1                               # DW_AT_decl_file
	.byte	2                               # DW_AT_decl_line
	.byte	2                               # DW_AT_location
	.byte	251
	.byte	0
	.byte	5                               # Abbrev [5] 0x24:0x9 DW_TAG_structure_type
                                        # DW_AT_declaration
	.quad	-4149699470930386446            # DW_AT_signature
	.byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end1:
	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                               # Abbreviation Code
	.byte	65                              # DW_TAG_type_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	54                              # DW_AT_calling_convention
	.byte	11                              # DW_FORM_data1
	.byte	3                               # DW_AT_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	11                              # DW_AT_byte_size
	.byte	11                              # DW_FORM_data1
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	1                               # DW_CHILDREN_yes
	.byte	37                              # DW_AT_producer
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	19                              # DW_AT_language
	.byte	5                               # DW_FORM_data2
	.byte	3                               # DW_AT_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.ascii	"\260B"                         # DW_AT_GNU_dwo_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.ascii	"\261B"                         # DW_AT_GNU_dwo_id
	.byte	7                               # DW_FORM_data8
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	52                              # DW_TAG_variable
	.byte	0                               # DW_CHILDREN_no
	.byte	3                               # DW_AT_name
	.ascii	"\202>"                         # DW_FORM_GNU_str_index
	.byte	73                              # DW_AT_type
	.byte	19                              # DW_FORM_ref4
	.byte	63                              # DW_AT_external
	.byte	25                              # DW_FORM_flag_present
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	59                              # DW_AT_decl_line
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	24                              # DW_FORM_exprloc
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	5                               # Abbreviation Code
	.byte	19                              # DW_TAG_structure_type
	.byte	0                               # DW_CHILDREN_no
	.byte	60                              # DW_AT_declaration
	.byte	25                              # DW_FORM_flag_present
	.byte	105                             # DW_AT_signature
	.byte	32                              # DW_FORM_ref_sig8
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_line.dwo,"e",@progbits
.Ltmp0:
	.long	.Ldebug_line_end0-.Ldebug_line_start0 # unit length
.Ldebug_line_start0:
	.short	4
	.long	.Lprologue_end0-.Lprologue_start0
.Lprologue_start0:
	.byte	1
	.byte	1
	.byte	1
	.byte	-5
	.byte	14
	.byte	1
	.byte	0
	.ascii	"test.cpp"
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
.Lprologue_end0:
.Ldebug_line_end0:
	.section	.debug_addr,"",@progbits
.Laddr_table_base0:
	.quad	v1
	.section	.debug_gnu_pubnames,"",@progbits
	.long	.LpubNames_end0-.LpubNames_start0 # Length of Public Names Info
.LpubNames_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	36                              # Compilation Unit Length
	.long	25                              # DIE offset
	.byte	32                              # Attributes: VARIABLE, EXTERNAL
	.asciz	"v1"                            # External Name
	.long	0                               # End Mark
.LpubNames_end0:
	.section	.debug_gnu_pubtypes,"",@progbits
	.long	.LpubTypes_end0-.LpubTypes_start0 # Length of Public Types Info
.LpubTypes_start0:
	.short	2                               # DWARF Version
	.long	.Lcu_begin0                     # Offset of Compilation Unit Info
	.long	36                              # Compilation Unit Length
	.long	36                              # DIE offset
	.byte	16                              # Attributes: TYPE, EXTERNAL
	.asciz	"t1"                            # External Name
	.long	0                               # End Mark
.LpubTypes_end0:
	.ident	"clang version 14.0.0 (git@github.com:llvm/llvm-project.git c9e46219f38da5c3fbfe41012173dc893516826e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.section	.debug_line,"",@progbits
.Lline_table_start0:
