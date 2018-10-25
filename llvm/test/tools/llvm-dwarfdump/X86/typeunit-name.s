# Demonstrate that -name works with type units.
# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.o
# RUN: llvm-dwarfdump -name=V4_type_unit -name=V5_split_type_unit %t.o | FileCheck %s
#
# The names should appear twice, once for the unit and once for the type DIE,
# because we give them the same name.
# CHECK: V4_type_unit
# CHECK: V4_type_unit
# CHECK: V5_split_type_unit
# CHECK: V5_split_type_unit

        .section .debug_str,"MS",@progbits,1
str_TU_4:
        .asciz "V4_type_unit"

        .section .debug_str.dwo,"MS",@progbits,1
dwo_TU_5:
        .asciz "V5_split_type_unit"

# Abbrev section for the normal type unit.
        .section .debug_abbrev,"",@progbits
        .byte 0x01  # Abbrev code
        .byte 0x41  # DW_TAG_type_unit
        .byte 0x01  # DW_CHILDREN_yes
        .byte 0x03  # DW_AT_name
        .byte 0x0e  # DW_FORM_strp
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x02  # Abbrev code
        .byte 0x13  # DW_TAG_structure_type
        .byte 0x00  # DW_CHILDREN_no (no members)
        .byte 0x03  # DW_AT_name
        .byte 0x0e  # DW_FORM_strp
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x00  # EOM(3)

# And a .dwo copy for the .dwo section.
        .section .debug_abbrev.dwo,"",@progbits
        .byte 0x01  # Abbrev code
        .byte 0x41  # DW_TAG_type_unit
        .byte 0x01  # DW_CHILDREN_yes
        .byte 0x03  # DW_AT_name
        .byte 0x0e  # DW_FORM_strp
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x02  # Abbrev code
        .byte 0x13  # DW_TAG_structure_type
        .byte 0x00  # DW_CHILDREN_no (no members)
        .byte 0x03  # DW_AT_name
        .byte 0x0e  # DW_FORM_strp
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x00  # EOM(3)

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
        .byte 1
        .long str_TU_4
# The type DIE, which has the same name.
TU_4_type:
        .byte 2
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
        .byte 1
        .long dwo_TU_5
# The type DIE, which has the same name.
TU_split_5_type:
        .byte 2
        .long dwo_TU_5
        .byte 0 # NULL
        .byte 0 # NULL
TU_split_5_end:
