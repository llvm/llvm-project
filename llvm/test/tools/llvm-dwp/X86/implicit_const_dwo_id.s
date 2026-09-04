# Check that llvm-dwp reads DW_AT_GNU_dwo_id from the abbreviation table when it
# is encoded with DW_FORM_implicit_const (the value is stored inline in the
# abbreviation, not in .debug_info). The dwo_id must round-trip into the
# .debug_cu_index signature.

# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.o -split-dwarf-file=%t.dwo -dwarf-version=4
# RUN: llvm-dwp %t.dwo -o %t.dwp
# RUN: llvm-dwarfdump -debug-info -debug-cu-index %t.dwp | FileCheck %s

# CHECK: .debug_info.dwo contents:
# CHECK: DW_AT_GNU_dwo_id{{.*}}(305419896)

# CHECK: .debug_cu_index contents:
# CHECK: version = 2, units = 1, slots = 2
# CHECK: 0x0000000012345678 [0x0000000000000000

	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	4                       # DWARF version number
	.long	0                       # Offset Into Abbrev. Section
	.byte	8                       # Address Size (in bytes)
	.byte	1                       # Abbrev [1] DW_TAG_compile_unit
	                                # DW_AT_GNU_dwo_id has no value here; it is
	                                # carried inline by DW_FORM_implicit_const.
.Ldebug_info_dwo_end0:

	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	0                       # DW_CHILDREN_no
	.ascii	"\261B"                 # DW_AT_GNU_dwo_id (ULEB 0x2161)
	.byte	33                      # DW_FORM_implicit_const
	.sleb128 305419896              # implicit const value (0x12345678)
	.byte	0                       # EOM(1)
	.byte	0                       # EOM(2)
	.byte	0                       # EOM(3)
