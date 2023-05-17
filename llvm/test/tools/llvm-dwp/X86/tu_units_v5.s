# This test checks if llvm-dwp can correctly generate the tu index section (v5).

# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.o \
# RUN:         -split-dwarf-file=%t.dwo -dwarf-version=5
# RUN: llvm-dwp %t.dwo -o %t.dwp
# RUN: llvm-dwarfdump -debug-info -debug-tu-index %t.dwp | FileCheck %s

## Note: In order to check whether the type unit index is generated
## there is no need to add the missing DIEs for the structure type of the type unit.

# CHECK-DAG: .debug_info.dwo contents:
# CHECK: 0x00000000: Type Unit: length = 0x00000017, format = DWARF32, version = 0x0005, unit_type = DW_UT_split_type, abbr_offset = 0x0000, addr_size = 0x08, name = '', type_signature = [[TUID1:.*]], type_offset = 0x0019 (next unit at 0x0000001b)
# CHECK: 0x0000001b: Type Unit: length = 0x00000017, format = DWARF32, version = 0x0005, unit_type = DW_UT_split_type, abbr_offset = 0x0000, addr_size = 0x08, name = '', type_signature = [[TUID2:.*]], type_offset = 0x0019 (next unit at 0x00000036)
# CHECK-DAG: .debug_tu_index contents:
# CHECK: version = 5, units = 2, slots = 4
# CHECK: Index Signature          INFO                                     ABBREV
# CHECK:     1 [[TUID1]]          [0x0000000000000000, 0x000000000000001b) [0x00000000, 0x00000010)
# CHECK:     4 [[TUID2]]          [0x000000000000001b, 0x0000000000000036) [0x00000000, 0x00000010)

    .section	.debug_info.dwo,"e",@progbits
    .long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
    .short	5                               # DWARF version number
    .byte	6                               # DWARF Unit Type (DW_UT_split_type)
    .byte	8                               # Address Size (in bytes)
    .long	0                               # Offset Into Abbrev. Section
    .quad	5657452045627120676             # Type Signature
    .long	25                              # Type DIE Offset
    .byte	2                               # Abbrev [2] DW_TAG_type_unit
    .byte	3                               # Abbrev [3] DW_TAG_structure_type
    .byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end0:
    .section	.debug_info.dwo,"e",@progbits
    .long	.Ldebug_info_dwo_end1-.Ldebug_info_dwo_start1 # Length of Unit
.Ldebug_info_dwo_start1:
    .short	5                               # DWARF version number
    .byte	6                               # DWARF Unit Type (DW_UT_split_type)
    .byte	8                               # Address Size (in bytes)
    .long	0                               # Offset Into Abbrev. Section
    .quad	-8528522068957683993            # Type Signature
    .long	25                              # Type DIE Offset
    .byte	4                               # Abbrev [4] DW_TAG_type_unit
    .byte	5                               # Abbrev [5] DW_TAG_structure_type
    .byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end1:
    .section	.debug_info.dwo,"e",@progbits
    .long	.Ldebug_info_dwo_end2-.Ldebug_info_dwo_start2 # Length of Unit
.Ldebug_info_dwo_start2:
    .short	5                               # DWARF version number
    .byte	5                               # DWARF Unit Type (DW_UT_split_compile)
    .byte	8                               # Address Size (in bytes)
    .long	0                               # Offset Into Abbrev. Section
    .quad	0
    .byte	1                               # Abbrev [1] DW_TAG_compile_unit
.Ldebug_info_dwo_end2:
    .section	.debug_abbrev.dwo,"e",@progbits
    .byte	1                               # Abbreviation Code
    .byte	17                              # DW_TAG_compile_unit
    .byte	0                               # DW_CHILDREN_no
    .byte	0                               # EOM(1)
    .byte	0                               # EOM(2)
    .byte	2                               # Abbreviation Code
    .byte	65                              # DW_TAG_type_unit
    .byte	1                               # DW_CHILDREN_yes
    .byte	0                               # EOM
    .byte	0                               # EOM
    .byte	4                               # Abbreviation Code
    .byte	65                              # DW_TAG_type_unit
    .byte	1                               # DW_CHILDREN_yes
    .byte	0                               # EOM
    .byte	0                               # EOM
    .byte	0                               # EOM
