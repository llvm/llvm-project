# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o %t.o
# RUN: llvm-dwarfdump %t.o | FileCheck %s

# CHECK: 0x00000000: Compile Unit: length = 0x00000009, format = DWARF32, version = 0x0005, unit_type = DW_UT_compile, abbr_offset = 0x0000, addr_size = 0x08 (next unit at 0x0000000d)
# CHECK: 0x0000000c: DW_TAG_compile_unit

	.section	.debug_abbrev,"",@progbits
	.byte	1                       # Abbreviation Code
	.byte	17                      # DW_TAG_compile_unit
	.byte	0                       # DW_CHILDREN_no
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
	.byte	1                       # Abbrev [1] 0xc:0x22 DW_TAG_compile_unit
.Ldebug_info_end0:
	.section	.debug_loclists,"",@progbits
	.byte   0
