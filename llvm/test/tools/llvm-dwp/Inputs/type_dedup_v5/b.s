## Note: For the purpose of checking the de-duplication of type units
## it is not necessary to have the DIEs for the structure type, that
## are referenced by the type unit.

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
    .long	.Ldebug_info_dwo_end2-.Ldebug_info_dwo_start2 # Length of Unit
.Ldebug_info_dwo_start2:
    .short	5                               # DWARF version number
    .byte	5                               # DWARF Unit Type (DW_UT_split_compile)
    .byte	8                               # Address Size (in bytes)
    .long	0                               # Offset Into Abbrev. Section
    .quad	-1709724327721109161
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
    .byte	0                               # EOM
