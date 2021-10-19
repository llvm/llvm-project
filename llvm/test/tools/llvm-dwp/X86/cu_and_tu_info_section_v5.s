# This test checks if llvm-dwp can find the compilation unit if
# both type and compile units are available in the debug info section (v5)

# RUN: llvm-mc --triple=x86_64-unknown-linux --filetype=obj --split-dwarf-file=%t.dwo -dwarf-version=5 %s -o %t.o
# RUN: llvm-dwp %t.dwo -o %t.dwp
# RUN: llvm-dwarfdump -debug-info -debug-tu-index %t.dwp | FileCheck %s

## Note: For this test we do not need to define the DIE for the structure type, as we only want to
## have the info on the type and compile units.

# CHECK-DAG: .debug_info.dwo contents
# CHECK: 0x00000000: Type Unit: length = 0x00000017, format = DWARF32, version = 0x0005, unit_type = DW_UT_split_type, abbr_offset = 0x0000, addr_size = 0x08, name = '', type_signature = {{.*}}, type_offset = 0x0019 (next unit at 0x0000001b)
# CHECK: 0x0000001b: Compile Unit: length = 0x00000011, format = DWARF32, version = 0x0005, unit_type = DW_UT_split_compile, abbr_offset = 0x0000, addr_size = 0x08, DWO_id = {{.*}} (next unit at 0x00000030)
    .section	.debug_info.dwo,"e",@progbits
    .long	.Ldebug_info_dwo_end0-.Ldebug_info_dwo_start0 # Length of Unit
.Ldebug_info_dwo_start0:
    .short	5                               # DWARF version number
    .byte	6                               # DWARF Unit Type (DW_UT_split_type)
    .byte	8                               # Address Size (in bytes)
    .long	0                               # Offset Into Abbrev. Section
    .quad	5657452045627120676             # Type Signature
    .long	25                              # Type DIE Offset
    .byte	1                               # Abbrev [1] DW_TAG_type_unit
    .byte	2                               # Abbrev [2] DW_TAG_structure_type
    .byte	0                               # End Of Children Mark
.Ldebug_info_dwo_end0:
    .section	.debug_info.dwo,"e",@progbits
    .long	.Ldebug_info_dwo_end1-.Ldebug_info_dwo_start1 # Length of Unit
.Ldebug_info_dwo_start1:
    .short	5                               # DWARF version number
    .byte	5                               # DWARF Unit Type (DW_UT_split_compile)
    .byte	8                               # Address Size (in bytes)
    .long	0                               # Offset Into Abbrev. Section
    .quad	-1506010254921578184
    .byte	3                               # Abbrev [3] DW_TAG_compile_unit
.Ldebug_info_dwo_end1:
    .section	.debug_abbrev.dwo,"e",@progbits
    .byte	1                               # Abbreviation Code
    .byte	65                              # DW_TAG_type_unit
    .byte	1                               # DW_CHILDREN_yes
    .byte	0                               # EOM(1)
    .byte	0                               # EOM(2)
    .byte	3                               # Abbreviation Code
    .byte	17                              # DW_TAG_compile_unit
    .byte	0                               # DW_CHILDREN_no
    .byte	0                               # EOM(1)
    .byte	0                               # EOM(2)
    .byte	0                               # EOM(3)
