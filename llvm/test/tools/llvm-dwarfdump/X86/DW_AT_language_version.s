# Demonstrate dumping DW_AT_language_version.
# RUN: llvm-mc -triple=x86_64--linux -filetype=obj < %s | \
# RUN:     llvm-dwarfdump -v - | FileCheck %s

# CHECK: .debug_abbrev contents:
# CHECK: DW_AT_language_version DW_FORM_data4
# CHECK: DW_AT_language_version DW_FORM_data2
# CHECK: .debug_info contents:
# CHECK: DW_AT_language_version [DW_FORM_data4] (201402)
# CHECK: DW_AT_language_version [DW_FORM_data2] (0)

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_no 
        .ascii  "\221\001"              # DW_AT_language_version
        .byte   6                       # DW_FORM_data4
        .ascii  "\221\001"              # DW_AT_language_version
        .byte   5                       # DW_FORM_data2
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
        .long   201402                  # DW_AT_language_version
        .short  0                       # DW_AT_language_version
        .byte   0 
.Ldebug_info_end0:
