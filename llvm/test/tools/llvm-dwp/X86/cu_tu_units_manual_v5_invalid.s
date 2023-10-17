# This test checks that llvm-dwarfdump correctly reports errors when parsing
# DWARF Unit Headers in DWP files

# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.o \
# RUN:         -split-dwarf-file=%t.dwo -dwarf-version=5
# RUN: llvm-dwp %t.dwo -o %t.dwp
# RUN: llvm-dwarfdump -debug-info -debug-cu-index -debug-tu-index \
# RUN:                -manaully-generate-unit-index %t.dwp 2>&1 | FileCheck %s

## Note: In order to check whether the type unit index is generated
## there is no need to add the missing DIEs for the structure type of the type unit.

# CHECK-NOT: .debug_info.dwo contents:

# CHECK-DAG: .debug_cu_index contents:
# CHECK: Failed to parse CU header in DWP file
# CHECK-NEXT: DWARF unit at offset 0x00000000 has unsupported version 6, supported are 2-5

# CHECK-DAG: .debug_tu_index contents:
# CHECK: Failed to parse CU header in DWP file
# CHECK-NEXT: DWARF unit at offset 0x00000000 has unsupported version 6, supported are 2-5

    .section	.debug_info.dwo,"e",@progbits
    .long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
    .short	6                               # DWARF version number
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
    .short	6                               # DWARF version number
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
    .short	6                               # DWARF version number
    .byte	5                               # DWARF Unit Type (DW_UT_split_compile)
    .byte	8                               # Address Size (in bytes)
    .long	0                               # Offset Into Abbrev. Section
    .quad	1152943841751211454
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
