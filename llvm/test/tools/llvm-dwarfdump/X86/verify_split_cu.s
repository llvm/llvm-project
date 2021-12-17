# RUN: llvm-mc %s -filetype obj -triple x86_64-linux-gnu -o - \
# RUN: | not llvm-dwarfdump -v -verify - 2>&1 \
# RUN: | FileCheck %s --implicit-check-not=error --implicit-check-not=warning

# CHECK: Verifying dwo Units...
# CHECK: error: Compilation unit root DIE is not a unit DIE: DW_TAG_null.
# CHECK: error: Compilation unit type (DW_UT_split_compile) and root DIE (DW_TAG_null) do not match.
# CHECK: error: Unsupported DW_AT_location encoding: DW_FORM_data1
# FIXME: This should read "type unit" or just "unit" to be correct for this case/in general
# CHECK: error: DIE has DW_AT_decl_file that references a file with index 1 and the compile unit has no line table
# CHECK: Errors detected
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end1-.Ldebug_info_dwo_start1 # Length of Unit
.Ldebug_info_dwo_start1:
	.short	5                               # DWARF version number
	.byte	5                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	0                               # Offset Into Abbrev. Section
	.quad	1                               # DWO ID
	.quad   0
.Ldebug_info_dwo_end1:
	.long	.Ldebug_info_dwo_end2-.Ldebug_info_dwo_start2 # Length of Unit
.Ldebug_info_dwo_start2:
	.short	5                               # DWARF version number
	.byte	5                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	0                               # Offset Into Abbrev. Section
	.quad	2                               # DWO ID
	.byte	1                               # Abbrev [1] DW_TAG_compile_unit
	.byte	0                               # DW_AT_decl_file
	.byte	0                               # DW_AT_location
.Ldebug_info_dwo_end2:
.Ldebug_info_dwo_prestart3:
	.long	.Ldebug_info_dwo_end3-.Ldebug_info_dwo_start3 # Length of Unit
.Ldebug_info_dwo_start3:
	.short	5                               # DWARF version number
	.byte	6                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	0                               # Offset Into Abbrev. Section
	.quad	3                               # Type Signature
	.long	.Ldebug_info_dwo_die3-.Ldebug_info_dwo_prestart3 # Type DIE Offset
.Ldebug_info_dwo_die3:
	.byte	2                               # Abbrev [2] DW_TAG_type_unit
	.byte	1                               # DW_AT_decl_file
.Ldebug_info_dwo_end3:
.Ldebug_info_dwo_prestart4:
	.long	.Ldebug_info_dwo_end4-.Ldebug_info_dwo_start4 # Length of Unit
.Ldebug_info_dwo_start4:
	.short	5                               # DWARF version number
	.byte	6                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	0                               # Offset Into Abbrev. Section
	.quad	4                               # Type Signature
	.long	.Ldebug_info_dwo_die4-.Ldebug_info_dwo_prestart4 # Type DIE Offset
.Ldebug_info_dwo_die4:
	.byte	3                               # Abbrev [3] DW_TAG_type_unit
	.long	0                               # DW_AT_stmt_list
	.byte	0                               # DW_AT_decl_file
.Ldebug_info_dwo_end4:
	.long	.Ldebug_info_dwo_end5-.Ldebug_info_dwo_start5 # Length of Unit
.Ldebug_info_dwo_start5:
	.short	5                               # DWARF version number
	.byte	5                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	0                               # Offset Into Abbrev. Section
	.quad	2                               # DWO ID
	.byte	4                               # Abbrev [1] DW_TAG_compile_unit
	.uleb128	0                       # DW_AT_location
.Ldebug_info_dwo_end5:
	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	2                               # DW_AT_location
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	2                               # Abbreviation Code
	.byte	65                              # DW_TAG_type_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	3                               # Abbreviation Code
	.byte	65                              # DW_TAG_type_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	16                              # DW_AT_stmt_list
	.byte	23                              # DW_FORM_sec_offset
	.byte	58                              # DW_AT_decl_file
	.byte	11                              # DW_FORM_data1
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	4                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	2                               # DW_AT_location
	.byte	34                              # DW_FORM_loclistx
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
	.section	.debug_line.dwo,"e",@progbits
.Ltmp2:
	.long	.Ldebug_line_end0-.Ldebug_line_start0 # unit length
.Ldebug_line_start0:
	.short	5
	.byte	8
	.byte	0
	.long	.Lprologue_end0-.Lprologue_start0
.Lprologue_start0:
	.byte	1
	.byte	1
	.byte	1
	.byte	-5
	.byte	14
	.byte	1
	.byte	1
	.byte	1
	.byte	8
	.byte	1
	.ascii	"/usr/local/google/home/blaikie/dev/scratch"
	.byte	0
	.byte	3
	.byte	1
	.byte	8
	.byte	2
	.byte	15
	.byte	5
	.byte	30
	.byte	1
	.ascii	"test.cpp"
	.byte	0
	.byte	0
	.byte	0x5e, 0xc4, 0x4d, 0x3b
	.byte	0x78, 0xd0, 0x2a, 0x57
	.byte	0xd2, 0x75, 0xc1, 0x22
	.byte	0x36, 0xb7, 0x17, 0xbf
.Lprologue_end0:
.Ldebug_line_end0:
	.section	.debug_loclists.dwo,"e",@progbits
	.long	.Ldebug_list_header_end0-.Ldebug_list_header_start0 # Length
.Ldebug_list_header_start0:
	.short	5                               # Version
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
	.long	1                               # Offset entry count
.Lloclists_table_base0:
	.long	.Ldebug_loc0-.Lloclists_table_base0
.Ldebug_loc0:
	.byte	1                               # DW_LLE_base_addressx
	.byte	0                               #   base address index
	.byte	4                               # DW_LLE_offset_pair
	.uleb128 0                              #   starting offset
	.uleb128 1                              #   ending offset
	.byte	3                               # Loc expr size
	.byte	17                              # DW_OP_consts
	.byte	3                               # 3
	.byte	159                             # DW_OP_stack_value
	.byte	0                               # DW_LLE_end_of_list
.Ldebug_list_header_end0:
