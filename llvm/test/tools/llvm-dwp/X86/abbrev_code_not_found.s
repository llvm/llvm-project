# Check that llvm-dwp reports a clean error (instead of looping forever) when a
# compile unit references an abbreviation code that is absent from the
# abbreviation table.

# RUN: llvm-mc --triple=x86_64-unknown-linux --filetype=obj --split-dwarf-file=%t.dwo -dwarf-version=5 %s -o %t.o
# RUN: not llvm-dwp %t.dwo -o %t.dwp 2>&1 | FileCheck %s

# CHECK: abbrev code 2 not found in abbrev section

	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	5                       # DWARF version number
	.byte	5                       # DWARF Unit Type (DW_UT_split_compile)
	.byte	8                       # Address Size (in bytes)
	.long	0                       # Offset Into Abbrev. Section
	.quad	0x1100001122334455      # DWO_id
	.byte	2                       # Abbrev [2] -- not present in the table below
.Ldebug_info_dwo_end0:

	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	8                       # DW_FORM_string
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3) -- end of abbreviation table
