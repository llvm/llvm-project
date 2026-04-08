# Demonstrate dumping DW_AT_language_name.
# RUN: llvm-mc -triple=x86_64--linux -filetype=obj < %s | \
# RUN:     llvm-dwarfdump -v - | FileCheck %s

# CHECK: .debug_abbrev contents:
# CHECK: DW_AT_language_name DW_FORM_data2
# CHECK: DW_AT_language_name DW_FORM_data2
# CHECK: .debug_info contents:
# CHECK: DW_AT_language_name [DW_FORM_data2] (DW_LNAME_C)
# CHECK: DW_AT_language_name [DW_FORM_data2] (0x0000)

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_no 
        .ascii  "\220\001"              # DW_AT_language_name
        .byte   5                       # DW_FORM_data2
        .ascii  "\220\001"              # DW_AT_language_name
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
        .short  3                       # DW_AT_language_name
        .short  0                       # DW_AT_language_name
        .byte   0 
.Ldebug_info_end0:
