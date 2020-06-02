# this checks llvm-dwp handling of DWARFv5 Info section header.

# RUN: llvm-mc --triple=x86_64-unknown-linux --filetype=obj --split-dwarf-file=%t.dwo -dwarf-version=5 %s -o %t.o

# RUN: llvm-dwp %t.dwo -o %t.dwp
# RUN: llvm-dwarfdump -v %t.dwp | FileCheck %s

#CHECK-DAG: .debug_info.dwo contents:
#CHECK: 0x00000000: Compile Unit: length = 0x00000050, format = DWARF32, version = 0x0005, unit_type = DW_UT_split_compile, abbr_offset = 0x0000, addr_size = 0x08, DWO_id = [[DWOID:.*]] (next unit at 0x00000054)

# CHECK-DAG: .debug_cu_index contents:
# CHECK: version = 2 slots = 2
# CHECK: Index Signature          INFO                     ABBREV
# CHECK: 1 [[DWOID]] [0x00000000, 0x00000054) [0x00000000, 0x0000002a)

	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	5                      # DWARF version number
	.byte	5                       # DWARF Unit Type
	.byte	8                       # Address Size (in bytes)
	.long	0                       # Offset Into Abbrev. Section
	.quad	-1173350285159172090
	.byte	1                       # Abbrev [1] 0x14:0x16 DW_TAG_compile_unit
	.asciz  "clang version 11.0.0" # DW_AT_producer
	.short	12                     # DW_AT_language
	.asciz  "int.c"                # DW_AT_name
	.asciz  "int.dwo"              # DW_AT_dwo_name
	.byte	2                       # Abbrev [2] 0x1a:0xb DW_TAG_variable
	.asciz  "integer"              # DW_AT_name
	.long	37                      # DW_AT_type
                                        # DW_AT_external
	.byte	0                       # DW_AT_decl_file
	.byte	1                       # DW_AT_decl_line
	.byte	2                       # DW_AT_location
	.byte	161
	.byte	0
	.byte	3                       # Abbrev [3] 0x25:0x4 DW_TAG_base_type
	.asciz  "int"                  # DW_AT_name
	.byte	5                       # DW_AT_encoding
	.byte	4                       # DW_AT_byte_size
	.byte	0                       # End Of Children Mark
.Ldebug_info_dwo_end0:
	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	1                       # DW_CHILDREN_yes
	.byte	37                      # DW_AT_producer
	.byte	8                       # DW_FORM_string
	.byte	19                      # DW_AT_language
	.byte	5                       # DW_FORM_data2
	.byte	3                       # DW_AT_name
	.byte	8                       # DW_FORM_string
	.byte	118                     # DW_AT_dwo_name
	.byte	8                       # DW_FORM_string
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	2                       # Abbreviation Code
	.byte	52                      # DW_TAG_variable
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	8                       # DW_FORM_string
	.byte	73                      # DW_AT_type
	.byte	19                      # DW_FORM_ref4
	.byte	63                      # DW_AT_external
	.byte	25                      # DW_FORM_flag_present
	.byte	58                      # DW_AT_decl_file
	.byte	11                      # DW_FORM_data1
	.byte	59                      # DW_AT_decl_line
	.byte	11                      # DW_FORM_data1
	.byte	2                       # DW_AT_location
	.byte	24                      # DW_FORM_exprloc
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	3                       # Abbreviation Code
	.byte	36                      # DW_TAG_base_type
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	8                       # DW_FORM_string
	.byte	62                      # DW_AT_encoding
	.byte	11                      # DW_FORM_data1
	.byte	11                      # DW_AT_byte_size
	.byte	11                      # DW_FORM_data1
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
