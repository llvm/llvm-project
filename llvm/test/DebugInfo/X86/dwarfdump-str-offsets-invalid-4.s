# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.o
# RUN: llvm-dwarfdump -v %t.o | FileCheck --check-prefix=INVALIDLENGTH %s
#
# Test object to verify that llvm-dwarfdump handles an invalid string offsets
# table.

        .section .debug_str,"MS",@progbits,1
str_producer:
        .asciz "Handmade DWARF producer"
str_CU1:
        .asciz "Compile_Unit_1"

# A rudimentary abbrev section.
        .section .debug_abbrev,"",@progbits
        .byte 0x01  # Abbrev code
        .byte 0x11  # DW_TAG_compile_unit
        .byte 0x00  # DW_CHILDREN_no
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x00  # EOM(3)

# A rudimentary compile unit to convince dwarfdump that we are dealing with a
# DWARF v5 string offsets table.
        .section .debug_info,"",@progbits

# DWARF v5 CU header.
        .long  CU1_5_end-CU1_5_version  # Length of Unit
CU1_5_version:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev    # Offset Into Abbrev. Section
# A compile-unit DIE, which has no attributes.
        .byte 1                # Abbreviation code
CU1_5_end:

# Every unit contributes to the string_offsets table.
        .section .debug_str_offsets,"",@progbits
# CU1's contribution
# The length is not a multiple of 4. Check that we don't read off the
# end.
        .long .debug_str_offsets_segment0_end-.debug_str_offsets_base0
        .short 5    # DWARF version
        .short 0    # Padding
.debug_str_offsets_base0:
        .long str_producer
        .long str_CU1
        .byte 0
.debug_str_offsets_segment0_end:

# INVALIDLENGTH:             .debug_str_offsets contents:
# INVALIDLENGTH-NOT:         contents:
# INVALIDLENGTH:             error: contribution to string offsets table in section .debug_str_offsets has invalid length.
