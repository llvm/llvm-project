## Demonstrate dumping DW_AT_defaulted.
## Any ELF-target triple will work.
# RUN: llvm-mc -triple=x86_64--linux -filetype=obj < %s | \
# RUN:     llvm-dwarfdump -v - | FileCheck %s

# CHECK: .debug_abbrev contents:
# CHECK: DW_AT_defaulted DW_FORM_data1
# CHECK: .debug_info contents:
# CHECK: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_no)
# CHECK: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_in_class)
# CHECK: DW_AT_defaulted [DW_FORM_data1] (DW_DEFAULTED_out_of_class)

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_yes
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   2                       # Abbreviation Code
        .byte   46                      # DW_TAG_subprogram
        .byte   0                       # DW_CHILDREN_no
        .byte   0x8b, 1                 # DW_AT_defaulted (ULEB)
        .byte   11                      # DW_FORM_data1
        .byte   0                       # EOM(1)
        .byte   0                       # EOM(2)
        .byte   0                       # EOM(3)

        .section        .debug_info,"",@progbits
        .long   .Ldebug_info_end0-.Ldebug_info_start0 # Length of Unit
.Ldebug_info_start0:
        .short  5                       # DWARF version number
        .byte   1                       # Unit type
        .byte   8                       # Address Size (in bytes)
        .long   .debug_abbrev           # Offset Into Abbrev. Section
        .byte   1                       # Abbrev [1] DW_TAG_compile_unit
        .byte   2                       # Abbrev [2] DW_TAG_subprogram
        .byte   0                       # DW_DEFAULTED_no
        .byte   2                       # Abbrev [2] DW_TAG_subprogram
        .byte   1                       # DW_DEFAULTED_in_class
        .byte   2                       # Abbrev [2] DW_TAG_subprogram
        .byte   2                       # DW_DEFAULTED_out_of_class
        .byte   0                       # End Of Children Mark
.Ldebug_info_end0:
