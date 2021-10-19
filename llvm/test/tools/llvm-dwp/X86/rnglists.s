# This test checks if llvm-dwp outputs .debug_rnglists.

# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.o \
# RUN:         -split-dwarf-file=%t.dwo -dwarf-version=5
# RUN: llvm-dwp %t.dwo -o %t.dwp
# RUN: llvm-dwarfdump -debug-rnglists -debug-cu-index -debug-tu-index %t.dwp | FileCheck %s

# CHECK-DAG: .debug_cu_index contents:
# CHECK: Index Signature          INFO                     ABBREV                   RNGLISTS
# CHECK:     1 {{.*}} [0x00000018, 0x0000002d) [0x00000000, 0x00000004) [0x00000000, 0x00000017)

# CHECK-DAG: .debug_tu_index contents:
# CHECK: Index Signature          INFO                     ABBREV                   RNGLISTS
# CHECK:     2 {{.*}} [0x00000000, 0x00000018) [0x00000000, 0x00000004) [0x00000000, 0x00000017)

# CHECK-DAG: .debug_rnglists.dwo contents:
# range list header: length = 0x00000013, format = DWARF32, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000001
# CHECK: offsets: [
# CHECK-NEXT: 0x00000004
# CHECK-NEXT: ]
# CHECK-NEXT: ranges:
# CHECK-NEXT: [0x0000000000000004, 0x0000000000000008)
# CHECK-NEXT: [0x000000000000000c, 0x0000000000000010)

	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
	.short	5                               # DWARF version number
	.byte	6                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	0                               # Offset Into Abbrev. Section
	.quad	-4287463584810542331            # Type Signature
	.long	31                              # Type DIE Offset
.Ldebug_info_dwo_end0:
	.section	.debug_info.dwo,"e",@progbits
	.long	.Ldebug_info_dwo_end3-.Ldebug_info_dwo_start3 # Length of Unit
.Ldebug_info_dwo_start3:
	.short	5                               # DWARF version number
	.byte	5                               # DWARF Unit Type
	.byte	8                               # Address Size (in bytes)
	.long	0                               # Offset Into Abbrev. Section
	.quad	1152943841751211454
	.byte	1                              # Abbrev [1] 0x14:0x349 DW_TAG_compile_unit
.Ldebug_info_dwo_end3:
	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                              # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
	.section	.debug_rnglists.dwo,"e",@progbits
	.long	.Ldebug_list_header_end1-.Ldebug_list_header_start1 # Length
.Ldebug_list_header_start1:
	.short	5                               # Version
	.byte	8                               # Address size
	.byte	0                               # Segment selector size
	.long	1                               # Offset entry count
.Lrnglists_dwo_table_base0:
	.long	.Ldebug_ranges0-.Lrnglists_dwo_table_base0
.Ldebug_ranges0:
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 4                            #   starting offset
	.uleb128 8                            #   ending offset
	.byte	4                               # DW_RLE_offset_pair
	.uleb128 12                           #   starting offset
	.uleb128 16                           #   ending offset
	.byte	0                               # DW_RLE_end_of_list
.Ldebug_list_header_end1:
