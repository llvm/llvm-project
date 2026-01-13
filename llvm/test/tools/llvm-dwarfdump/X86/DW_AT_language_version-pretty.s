# Demonstrate dumping DW_AT_language_version in human-readable form.
# RUN: llvm-mc -triple=x86_64--linux -filetype=obj -o %t.o < %s
# RUN: llvm-dwarfdump %t.o -v | FileCheck %s --check-prefix=VERBOSE
# RUN: llvm-dwarfdump %t.o | FileCheck %s --check-prefix=NO-VERBOSE

# VERBOSE: .debug_info contents:
# VERBOSE: DW_AT_language_name [DW_FORM_data2] (DW_LNAME_C)
# VERBOSE: DW_AT_language_version [DW_FORM_data4] (201112 C11)
# VERBOSE: DW_AT_language_name [DW_FORM_data2] (0x0000)
# VERBOSE: DW_AT_language_version [DW_FORM_data4] (12 Unknown)

# NO-VERBOSE: .debug_info contents:
# NO-VERBOSE: DW_AT_language_name (DW_LNAME_C)
# NO-VERBOSE: DW_AT_language_version (C11)
# NO-VERBOSE: DW_AT_language_name (0x0000)
# NO-VERBOSE: DW_AT_language_version (Unknown)

        .section        .debug_abbrev,"",@progbits
        .byte   1                       # Abbreviation Code
        .byte   17                      # DW_TAG_compile_unit
        .byte   1                       # DW_CHILDREN_no 
        .ascii  "\220\001"              # DW_AT_language_name
        .byte   5                       # DW_FORM_data2
        .ascii  "\221\001"              # DW_AT_language_version
        .byte   6                       # DW_FORM_data4
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
        .long   201112                  # DW_AT_language_version
        .byte   1                       # Abbrev [1] DW_TAG_compile_unit
        .short  0                       # DW_AT_language_name
        .long   12                      # DW_AT_language_version
        .byte   0 
.Ldebug_info_end0:
