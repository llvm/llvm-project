# RUN: llvm-mc --triple=x86_64-unknown-linux --filetype=obj --split-dwarf-file=%t.dwo -dwarf-version=5 %s -o %t.o
# RUN: not llvm-dwp %t.dwo -o /dev/null 2>&1 | FileCheck %s

# CHECK: error: no compile unit found in file: {{.*}}no_cu_found.s
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	5                      # DWARF version number
	.byte	12                       # DWARF Unit Type (DW_TAG_string_type, wrong type)
	.byte	8                       # Address Size (in bytes)
	.long	0                       # Offset Into Abbrev. Section
	.quad	-1173350285159172090
	.byte	1                       # Abbrev [1] 0x14:0x16 DW_TAG_compile_unit
	.asciz  "clang version 11.0.0" # DW_AT_producer
	.short	12                     # DW_AT_language
	.asciz  "int.c"                # DW_AT_name
	.asciz  "int.dwo"              # DW_AT_dwo_name
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
