# RUN: llvm-mc %s -filetype obj -triple x86_64-linux-gnu -o - \
# RUN: | llvm-dwarfdump -verify - 2>&1 \
# RUN: | FileCheck %s --implicit-check-not=error --implicit-check-not=warning

# CHECK: Verifying dwo Units...
# CHECK: No errors.

	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end6-.Ldebug_info_dwo_start6 # Length of Unit
.Ldebug_info_dwo_start6:
	.short	4                               # DWARF version number
	.long	0                               # Offset Into Abbrev. Section
	.byte	8                               # Address Size (in bytes)
	.byte	1                               # Abbrev [4] DW_TAG_compile_unit
	.byte	1                               # DW_AT_ranges
.Ldebug_info_dwo_end6:
	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                               # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	85                              # DW_AT_ranges
	.byte	23                              # DW_FORM_sec_offset
	.byte	0                               # EOM(1)
	.byte	0                               # EOM(2)
	.byte	0                               # EOM(3)
