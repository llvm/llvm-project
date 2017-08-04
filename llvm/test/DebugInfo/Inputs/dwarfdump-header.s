# Test object to verify dwarfdump handles v4 and v5 CU/TU/line headers.
# We have a representative set of units: v4 CU, v5 CU, v4 TU, v5 split TU.
# We have v4 and v5 line-table headers.
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
        .byte 0x10  # DW_AT_stmt_list
        .byte 0x17  # DW_FORM_sec_offset
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
# The compile-unit DIE, with DW_AT_producer, DW_AT_name, DW_AT_stmt_list.
        .byte 1
        .long str_producer
        .long str_CU_4
        .long LH_4_start
        .byte 0 # NULL
CU_4_end:

# DWARF v5 normal CU header.
        .long  CU_5_end-CU_5_version  # Length of Unit
CU_5_version:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev    # Offset Into Abbrev. Section
# The compile-unit DIE, with DW_AT_producer, DW_AT_name, DW_AT_stmt_list.
        .byte 1
        .long str_producer
        .long str_CU_5
        .long LH_5_start
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

        .section .debug_line,"",@progbits
# DWARF v4 line-table header.
LH_4_start:
        .long   LH_4_end-LH_4_version   # Length of Unit
LH_4_version:
        .short  4               # DWARF version number
        .long   LH_4_header_end-LH_4_params     # Length of Prologue
LH_4_params:
        .byte   1               # Minimum Instruction Length
        .byte   1               # Maximum Operations per Instruction
        .byte   1               # Default is_stmt
        .byte   -5              # Line Base
        .byte   14              # Line Range
        .byte   13              # Opcode Base
        .byte   0               # Standard Opcode Lengths
        .byte   1
        .byte   1
        .byte   1
        .byte   1
        .byte   0
        .byte   0
        .byte   0
        .byte   1
        .byte   0
        .byte   0
        .byte   1
        # Directory table
        .asciz  "Directory4a"
        .asciz  "Directory4b"
        .byte   0
        # File table
        .asciz  "File4a"        # File name 1
        .byte   1               # Directory index 1
        .byte   0x41            # Timestamp 1
        .byte   0x42            # File Size 1
        .asciz  "File4b"        # File name 2
        .byte   0               # Directory index 2
        .byte   0x43            # Timestamp 2
        .byte   0x44            # File Size 2
        .byte   0               # End of list
LH_4_header_end:
        # Line number program, which is empty.
LH_4_end:

# DWARF v5 line-table header.
LH_5_start:
        .long   LH_5_end-LH_5_version   # Length of Unit
LH_5_version:
        .short  5               # DWARF version number
        .byte   8               # Address Size
        .byte   0               # Segment Selector Size
        .long   LH_5_header_end-LH_5_params     # Length of Prologue
LH_5_params:
        .byte   1               # Minimum Instruction Length
        .byte   1               # Maximum Operations per Instruction
        .byte   1               # Default is_stmt
        .byte   -5              # Line Base
        .byte   14              # Line Range
        .byte   13              # Opcode Base
        .byte   0               # Standard Opcode Lengths
        .byte   1
        .byte   1
        .byte   1
        .byte   1
        .byte   0
        .byte   0
        .byte   0
        .byte   1
        .byte   0
        .byte   0
        .byte   1
        # Directory table format
        .byte   1               # One element per directory entry
        .byte   1               # DW_LNCT_path
        .byte   0x08            # DW_FORM_string
        # Directory table entries
        .byte   2               # Two directories
        .asciz "Directory5a"
        .asciz "Directory5b"
        # File table format
        .byte   4               # Four elements per file entry
        .byte   1               # DW_LNCT_path
        .byte   0x08            # DW_FORM_string
        .byte   2               # DW_LNCT_directory_index
        .byte   0x0b            # DW_FORM_data1
        .byte   3               # DW_LNCT_timestamp
        .byte   0x0f            # DW_FORM_udata
        .byte   4               # DW_LNCT_size
        .byte   0x0f            # DW_FORM_udata
        # File table entries
        .byte   2               # Two files
        .asciz "File5a"
        .byte   1
        .byte   0x51
        .byte   0x52
        .asciz "File5b"
        .byte   2
        .byte   0x53
        .byte   0x54
LH_5_header_end:
        # Line number program, which is empty.
LH_5_end:
