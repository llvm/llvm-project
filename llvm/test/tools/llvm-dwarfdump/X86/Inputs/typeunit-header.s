# Test object with an artifically constructed type unit header to verify 
# that the length field is correctly used to verify the validity of the
# type_offset field.
#
# To generate the test object:
# llvm-mc -triple x86_64-unknown-linux typeunit-header.s -filetype=obj \
#         -o typeunit-header.elf-x86-64
#
# We only have an abbreviation for the type unit die which is all we need.
# Real type unit dies have quite different attributes of course, but we
# just need to demonstrate an issue with validating length, so we just give it
# a single visibility attribute.
        .section .debug_abbrev,"",@progbits
        .byte 0x01  # Abbrev code
        .byte 0x41  # DW_TAG_type_unit
        .byte 0x01  # DW_CHILDREN_yes
        .byte 0x17  # DW_AT_visibility
        .byte 0x0b  # DW_FORM_data1
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x02  # Abbrev code
        .byte 0x13  # DW_TAG_structure_type
        .byte 0x00  # DW_CHILDREN_no (no members)
        .byte 0x17  # DW_AT_visibility
        .byte 0x0b  # DW_FORM_data1
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x00  # EOM(3)
        
        .section .debug_types,"",@progbits
# DWARF v4 Type unit header - DWARF32 format.
TU_4_32_start:
        .long TU_4_32_end-TU_4_32_version  # Length of Unit
TU_4_32_version:
        .short 4               # DWARF version number
        .long .debug_abbrev    # Offset Into Abbrev. Section
        .byte 8                # Address Size (in bytes)
        .quad 0x0011223344556677 # Type Signature
        .long TU_4_32_type-TU_4_32_start # Type offset
# The type-unit DIE, which has just a visibility attribute.
        .byte 1                # Abbreviation code
        .byte 1                # DW_VIS_local
# The type DIE, which also just has a one-byte visibility attribute.
TU_4_32_type:
        .byte 2                # Abbreviation code
        .byte 1                # DW_VIS_local
        .byte 0 # NULL
        .byte 0 # NULL
TU_4_32_end:
