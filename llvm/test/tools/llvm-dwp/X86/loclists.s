# This test checks if llvm-dwp outputs .debug_loclists.

# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.o \
# RUN:         -split-dwarf-file=%t.dwo -dwarf-version=5
# RUN: llvm-dwp %t.dwo -o %t.dwp
# RUN: llvm-dwarfdump -debug-loclists -debug-cu-index -debug-tu-index %t.dwp | FileCheck %s

# CHECK-DAG: .debug_loclists.dwo contents:
# CHECK: locations list header: length = 0x00000019, format = DWARF32, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000001
# CHECK-NEXT: offsets: [
# CHECK-NEXT: 0x00000004
# CHECK-NEXT: ]
# CHECK: DW_LLE_base_addressx   (0x0000000000000000)
# CHECK-NEXT: DW_LLE_offset_pair     (0x0000000000000000, 0x0000000000000004): DW_OP_reg5 RDI
# CHECK-NEXT: DW_LLE_offset_pair     (0x0000000000000004, 0x0000000000000008): DW_OP_reg3 RBX

# CHECK-DAG: .debug_cu_index contents:
# CHECK: Index Signature          INFO                     ABBREV                   LOCLISTS
# CHECK:     1 {{.*}} [0x00000018, 0x0000002d) [0x00000000, 0x00000004) [0x00000000, 0x0000001d)

# CHECK-DAG: .debug_tu_index contents:
# CHECK: Index Signature          INFO                     ABBREV                   LOCLISTS
# CHECK:     2 {{.*}} [0x00000000, 0x00000018) [0x00000000, 0x00000004) [0x00000000, 0x0000001d)

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
	.uleb128 0                            #   starting offset
	.uleb128 4                            #   ending offset
	.byte	1                               # Loc expr size
	.byte	85                              # DW_OP_reg5
	.byte	4                               # DW_LLE_offset_pair
	.uleb128 4                            #   starting offset
	.uleb128 8                            #   ending offset
	.byte	1                               # Loc expr size
	.byte	83                              # DW_OP_reg3
	.byte	0                               # DW_LLE_end_of_list
.Ldebug_list_header_end0:
	.section	.debug_abbrev.dwo,"e",@progbits
	.byte	1                              # Abbreviation Code
	.byte	17                              # DW_TAG_compile_unit
	.byte	0                               # DW_CHILDREN_no
	.byte	0                               # EOM(1)
