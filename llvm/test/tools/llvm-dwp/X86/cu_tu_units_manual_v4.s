# This test checks if we can correctly parse manull cu and tu index for DWARF4.

# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.o \
# RUN:         -split-dwarf-file=%t.dwo -dwarf-version=4
# RUN: llvm-dwp %t.dwo -o %t.dwp
# RUN: llvm-dwarfdump -debug-info -debug-types -debug-cu-index -debug-tu-index %t.dwp | FileCheck %s
# RUN: llvm-dwarfdump -debug-info -debug-types -debug-cu-index -debug-tu-index -manually-generate-unit-index %t.dwp | FileCheck %s

## Note: In order to check whether the type unit index is generated
## there is no need to add the missing DIEs for the structure type of the type unit.

# CHECK-DAG: .debug_info.dwo contents:
# CHECK: 0x00000000: Compile Unit: length = 0x00000010, format = DWARF32, version = 0x0004, abbr_offset = 0x0000, addr_size = 0x08 (next unit at 0x00000014)
# CHECK:  DW_AT_GNU_dwo_id  ([[CUID1:.*]])
# CHECK-DAG: .debug_types.dwo contents:
# CHECK: 0x00000000: Type Unit: length = 0x00000016, format = DWARF32, version = 0x0004, abbr_offset = 0x0000, addr_size = 0x08, name = '', type_signature = [[TUID1:.*]], type_offset = 0x0019 (next unit at 0x0000001a)
# CHECK: 0x0000001a: Type Unit: length = 0x00000016, format = DWARF32, version = 0x0004, abbr_offset = 0x0000, addr_size = 0x08, name = '', type_signature = [[TUID2:.*]], type_offset = 0x0019 (next unit at 0x00000034)
# CHECK-DAG: .debug_cu_index contents:
# CHECK: version = 2, units = 1, slots = 2
# CHECK: Index Signature          INFO                                     ABBREV
# CHECK:     2 [[CUID1]]          [0x0000000000000000, 0x0000000000000014) [0x00000000, 0x00000013)
# CHECK-DAG: .debug_tu_index contents:
# CHECK: version = 2, units = 2, slots = 4
# CHECK: Index Signature          TYPES                                    ABBREV
# CHECK:     1 [[TUID1]]          [0x0000000000000000, 0x000000000000001a) [0x00000000, 0x00000013)
# CHECK:     4 [[TUID2]]          [0x000000000000001a, 0x0000000000000034) [0x00000000, 0x00000013)

    .section	.debug_types.dwo,"e",@progbits
    .long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
    .short	4                               # DWARF version number
    .long	0                               # Offset Into Abbrev. Section
    .byte	8                               # Address Size (in bytes)
    .quad	5657452045627120676             # Type Signature
    .long	25                              # Type DIE Offset
    .byte	2                               # Abbrev [2] DW_TAG_type_unit
    .byte	3                               # Abbrev [3] DW_TAG_structure_type
    .byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end0:
    .section	.debug_types.dwo,"e",@progbits
    .long	.Ldebug_info_dwo_end1-.Ldebug_info_dwo_start1 # Length of Unit
.Ldebug_info_dwo_start1:
    .short	4                               # DWARF version number
    .long	0                               # Offset Into Abbrev. Section
    .byte	8                               # Address Size (in bytes)
    .quad	-8528522068957683993            # Type Signature
    .long	25                              # Type DIE Offset
    .byte	2                               # Abbrev [2] DW_TAG_type_unit
    .byte	3                               # Abbrev [3] DW_TAG_structure_type
    .byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end1:
    .section	.debug_info.dwo,"e",@progbits
    .long	.Ldebug_info_dwo_end2-.Ldebug_info_dwo_start2 # Length of Unit
.Ldebug_info_dwo_start2:
    .short	4                               # DWARF version number
    .long	0                               # Offset Into Abbrev. Section
    .byte	8                               # Address Size (in bytes)
    .byte	1                               # Abbrev [1] DW_TAG_compile_unit
    .quad	-6619898858003450627            # DW_AT_GNU_dwo_id
.Ldebug_info_dwo_end2:
    .section	.debug_abbrev.dwo,"e",@progbits
    .byte	1                               # Abbreviation Code
    .byte	17                              # DW_TAG_compile_unit
    .byte	0                               # DW_CHILDREN_no
    .ascii	"\261B"                         # DW_AT_GNU_dwo_id
    .byte	7                               # DW_FORM_data8
    .byte	0                               # EOM(1)
    .byte	0                               # EOM(2)
    .byte	2                               # Abbreviation Code
    .byte	65                              # DW_TAG_type_unit
    .byte	1                               # DW_CHILDREN_yes
    .byte	0                               # EOM
    .byte	0                               # EOM
    .byte	3                               # Abbreviation Code
    .byte	0x13                            # DW_TAG_structure_unit
    .byte	0                               # DW_CHILDREN_no
    .byte	0                               # EOM
    .byte	0                               # EOM
    .byte	0                               # EOM
