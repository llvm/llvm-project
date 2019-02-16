# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.o
# RUN: not llvm-dwarfdump -verify %t.o | FileCheck --check-prefix=VERIFY %s

# Check that the verifier correctly diagnoses various error conditions with
# the usage of string indices and string offsets tables.

        .section .debug_str,"MS",@progbits,1
str_producer:
        .asciz "Handmade DWARF producer"

        .section .debug_str_offsets,"",@progbits
# The string offsets table
        .long .debug_str_offsets_segment0_end-.debug_str_offsets_base0+4
        .short 5    # DWARF version
        .short 0    # Padding
.debug_str_offsets_base0:
        .long str_producer
        .long 1000  # Invalid string address.
.debug_str_offsets_segment0_end:

# A simple abbrev section with a basic compile unit DIE.
        .section .debug_abbrev,"",@progbits
        .byte 0x01  # Abbrev code
        .byte 0x11  # DW_TAG_compile_unit
        .byte 0x01  # DW_CHILDREN_no
        .byte 0x25  # DW_AT_producer
        .byte 0x1a  # DW_FORM_strx
        .byte 0x72  # DW_AT_str_offsets_base
        .byte 0x17  # DW_FORM_sec_offset
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)

        .section .debug_info,"",@progbits

# The first unit's CU DIE has an invalid DW_AT_str_offsets_base which
# renders any string index unresolvable.

# DWARF v5 CU header.
        .long  CU1_5_end-CU1_5_version  # Length of Unit
CU1_5_version:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev    # Offset Into Abbrev. Section
# The compile-unit DIE, which has DW_AT_producer and DW_AT_str_offsets.
        .byte 1                # Abbreviation code
        .byte 0                # Index of string for DW_AT_producer.
        .long 1000             # Bad value for DW_AT_str_offsets_base
        .byte 0 # NULL
CU1_5_end:

# The second unit's CU DIE uses an invalid string index.

# DWARF v5 CU header
        .long  CU2_5_end-CU2_5_version  # Length of Unit
CU2_5_version:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev    # Offset Into Abbrev. Section
# The compile-unit DIE, which has DW_AT_producer and DW_AT_str_offsets.
        .byte 1                # Abbreviation code
        .byte 100              # Invalid string index
        .long .debug_str_offsets_base0
        .byte 0 # NULL
CU2_5_end:

# The third unit's CU DIE uses a valid string index but the entry in the 
# string offsets table is invalid. 

# DWARF v5 CU header
        .long  CU3_5_end-CU3_5_version  # Length of Unit
CU3_5_version:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev    # Offset Into Abbrev. Section
# The compile-unit DIE, which has DW_AT_producer and DW_AT_str_offsets.
        .byte 1                # Abbreviation code
        .byte 1                # Index of string for DW_AT_producer.
        .long .debug_str_offsets_base0
        .byte 0 # NULL
CU3_5_end:
        
# VERIFY-DAG:      error: DW_FORM_strx used without a valid string offsets table:
# VERIFY-DAG:      error: DW_FORM_strx uses index 100, which is too large:
# VERIFY-DAG:      error: DW_FORM_strx uses index 1, but the referenced string offset 
# VERIFY-DAG-SAME: is beyond .debug_str bounds:
