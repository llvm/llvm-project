# Test object to verify dwarfdump handles v4 and v5 CU/TU headers.
# We have a representative set of units: v4 CU, v5 CU, v4 TU, v5 split TU.
#
# To generate the test object:
# llvm-mc -triple x86_64-unknown-linux dwarfdump-header.s -filetype=obj \
#         -o dwarfdump-header.elf-x86-64

        .section .debug_str,"MS",@progbits,1
str_producer:
        .asciz "Handmade DWARF producer"
str_CU_4:
        .asciz "V4_compile_unit"
str_CU_5:
        .asciz "V5_compile_unit"
str_TU_4:
        .asciz "V4_type_unit"

        .section .debug_str.dwo,"MS",@progbits,1
dwo_TU_5:
        .asciz "V5_split_type_unit"

# All CUs/TUs use the same abbrev section for simplicity.
        .section .debug_abbrev,"",@progbits
        .byte 0x01  # Abbrev code
        .byte 0x11  # DW_TAG_compile_unit
        .byte 0x00  # DW_CHILDREN_no
        .byte 0x25  # DW_AT_producer
        .byte 0x0e  # DW_FORM_strp
        .byte 0x03  # DW_AT_name
        .byte 0x0e  # DW_FORM_strp
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x02  # Abbrev code
        .byte 0x41  # DW_TAG_type_unit
        .byte 0x01  # DW_CHILDREN_yes
        .byte 0x03  # DW_AT_name
        .byte 0x0e  # DW_FORM_strp
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x03  # Abbrev code
        .byte 0x13  # DW_TAG_structure_type
        .byte 0x00  # DW_CHILDREN_no (no members)
        .byte 0x03  # DW_AT_name
        .byte 0x0e  # DW_FORM_strp
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x00  # EOM(3)

# And a .dwo copy for the .dwo sections.
        .section .debug_abbrev.dwo,"",@progbits
        .byte 0x01  # Abbrev code
        .byte 0x11  # DW_TAG_compile_unit
        .byte 0x00  # DW_CHILDREN_no
        .byte 0x25  # DW_AT_producer
        .byte 0x0e  # DW_FORM_strp
        .byte 0x03  # DW_AT_name
        .byte 0x0e  # DW_FORM_strp
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x02  # Abbrev code
        .byte 0x41  # DW_TAG_type_unit
        .byte 0x01  # DW_CHILDREN_yes
        .byte 0x03  # DW_AT_name
        .byte 0x0e  # DW_FORM_strp
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x03  # Abbrev code
        .byte 0x13  # DW_TAG_structure_type
        .byte 0x00  # DW_CHILDREN_no (no members)
        .byte 0x03  # DW_AT_name
        .byte 0x0e  # DW_FORM_strp
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x00  # EOM(3)

        .section .debug_info,"",@progbits

# DWARF v4 CU header. V4 CU headers all look the same so we do only one.
        .long  CU_4_end-CU_4_version  # Length of Unit
CU_4_version:
        .short 4               # DWARF version number
        .long .debug_abbrev    # Offset Into Abbrev. Section
        .byte 8                # Address Size (in bytes)
# The compile-unit DIE, which has just DW_AT_producer and DW_AT_name.
        .byte 1
        .long str_producer
        .long str_CU_4
        .byte 0 # NULL
CU_4_end:

# DWARF v5 normal CU header.
        .long  CU_5_end-CU_5_version  # Length of Unit
CU_5_version:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev    # Offset Into Abbrev. Section
# The compile-unit DIE, which has just DW_AT_producer and DW_AT_name.
        .byte 1
        .long str_producer
        .long str_CU_5
        .byte 0 # NULL
CU_5_end:

        .section .debug_types,"",@progbits

# DWARF v4 Type unit header. Normal/split are identical so we do only one.
TU_4_start:
        .long  TU_4_end-TU_4_version  # Length of Unit
TU_4_version:
        .short 4               # DWARF version number
        .long .debug_abbrev    # Offset Into Abbrev. Section
        .byte 8                # Address Size (in bytes)
        .quad 0x0011223344556677 # Type Signature
        .long TU_4_type-TU_4_start # Type offset
# The type-unit DIE, which has a name.
        .byte 2
        .long str_TU_4
# The type DIE, which has a name.
TU_4_type:
        .byte 3
        .long str_TU_4
        .byte 0 # NULL
        .byte 0 # NULL
TU_4_end:

        .section .debug_types.dwo,"",@progbits
# FIXME: DWARF v5 wants type units in .debug_info[.dwo] not .debug_types[.dwo].

# DWARF v5 split type unit header.
TU_split_5_start:
        .long  TU_split_5_end-TU_split_5_version  # Length of Unit
TU_split_5_version:
        .short 5               # DWARF version number
        .byte 6                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev.dwo    # Offset Into Abbrev. Section
        .quad 0x8899aabbccddeeff # Type Signature
        .long TU_split_5_type-TU_split_5_start  # Type offset
# The type-unit DIE, which has a name.
        .byte 2
        .long dwo_TU_5
# The type DIE, which has a name.
TU_split_5_type:
        .byte 3
        .long dwo_TU_5
        .byte 0 # NULL
        .byte 0 # NULL
TU_split_5_end:
