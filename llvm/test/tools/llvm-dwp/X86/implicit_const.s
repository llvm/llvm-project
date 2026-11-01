# Check that llvm-dwp correctly handles a DWARFv5 .dwo whose abbreviation table
# uses DW_FORM_implicit_const. DW_FORM_implicit_const stores its value as an
# SLEB128 in the abbreviation declaration itself (not in .debug_info). If
# llvm-dwp fails to consume that value while scanning the abbreviation table, the
# (attribute, form) walk desyncs; here the desync makes the scan for the
# compile-unit's abbreviation code run off the end of .debug_abbrev and spin
# forever. This is a regression test: before the fix llvm-dwp hangs, after the
# fix it terminates and emits the expected CU identifiers.

# RUN: llvm-mc --triple=x86_64-unknown-linux --filetype=obj --split-dwarf-file=%t.dwo -dwarf-version=5 %s -o %t.o
# RUN: llvm-dwp %t.dwo -o %t.dwp
# RUN: llvm-dwarfdump -v %t.dwp | FileCheck %s

# CHECK: .debug_info.dwo contents:
# CHECK: 0x00000000: Compile Unit: {{.*}} version = 0x0005, unit_type = DW_UT_split_compile, {{.*}} DWO_id = [[DWOID:0x[0-9a-f]+]]
# CHECK: DW_TAG_compile_unit
# CHECK: DW_AT_name{{.*}}"test.c"
# CHECK: DW_AT_dwo_name{{.*}}"test.dwo"

# CHECK: .debug_cu_index contents:
# CHECK: version = 5, units = 1, slots = 2
# CHECK: [[DWOID]] [0x0000000000000000

	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	5                       # DWARF version number
	.byte	5                       # DWARF Unit Type (DW_UT_split_compile)
	.byte	8                       # Address Size (in bytes)
	.long	0                       # Offset Into Abbrev. Section
	.quad	0x1100001122334455      # DWO_id
	.byte	2                       # Abbrev [2] DW_TAG_compile_unit
	.asciz	"test.c"                # DW_AT_name
	.asciz	"test.dwo"              # DW_AT_dwo_name
.Ldebug_info_dwo_end0:

	.section	.debug_abbrev.dwo,"e",@progbits
	# Abbrev [1] DW_TAG_variable, declared *before* the compile unit and using
	# DW_FORM_implicit_const. This is the declaration that desyncs the walk.
	.byte	1                       # Abbreviation Code
	.byte	52                      # DW_TAG_variable
	.byte	0                       # DW_CHILDREN_no
	.byte	28                      # DW_AT_const_value
	.byte	33                      # DW_FORM_implicit_const
	.byte	42                      # implicit const value (SLEB128)
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	# Abbrev [2] DW_TAG_compile_unit
	.byte	2                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	0                       # DW_CHILDREN_no
	.byte	3                       # DW_AT_name
	.byte	8                       # DW_FORM_string
	.byte	118                     # DW_AT_dwo_name
	.byte	8                       # DW_FORM_string
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
