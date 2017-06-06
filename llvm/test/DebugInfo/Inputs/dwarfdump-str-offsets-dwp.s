# Test object to verify that dwarfdump handles dwp files with DWARF v5 string
# offset tables. We have 2 CUs and 2 TUs, where it is assumed that 
# CU1 and TU1 came from one object file, CU2 and TU2 from a second object
# file.
#
# To generate the test object:
# llvm-mc -triple x86_64-unknown-linux dwarfdump-str-offsets-dwp.s -filetype=obj \
#         -o dwarfdump-str_offsets-dwp.x86_64.o

        .section .debug_str.dwo,"MS",@progbits,1
str_producer:
        .asciz "Handmade DWARF producer"
str_CU1:
        .asciz "Compile_Unit_1"
str_CU1_dir:
        .asciz "/home/test/CU1"
str_CU2:
        .asciz "Compile_Unit_2"
str_CU2_dir:
        .asciz "/home/test/CU2"
str_TU1:
        .asciz "Type_Unit_1"
str_TU1_type:
        .asciz "MyStruct_1"
str_TU2:
        .asciz "Type_Unit_2"
str_TU2_type:
        .asciz "MyStruct_2"

        .section .debug_str_offsets.dwo,"",@progbits
# Object files 1's portion of the .debug_str_offsets.dwo section.
.debug_str_offsets_object_file1:

# CU1's contribution (from object file 1)
.debug_str_offsets_start_CU1:
        .long .debug_str_offsets_end_CU1-.debug_str_offsets_base_CU1
        .short 5    # DWARF version
        .short 0    # Padding
.debug_str_offsets_base_CU1:
        .long str_producer-.debug_str.dwo
        .long str_CU1-.debug_str.dwo
        .long str_CU1_dir-.debug_str.dwo
.debug_str_offsets_end_CU1:

# TU1's contribution (from object file 1)
.debug_str_offsets_start_TU1:
        .long .debug_str_offsets_end_TU1-.debug_str_offsets_base_TU1
        .short 5    # DWARF version
        .short 0    # Padding
.debug_str_offsets_base_TU1:
        .long str_TU1-.debug_str.dwo
        .long str_TU1_type-.debug_str.dwo
.debug_str_offsets_end_TU1:

# Object files 2's portion of the .debug_str_offsets.dwo section.
.debug_str_offsets_object_file2:

# CU2's contribution (from object file 2)
.debug_str_offsets_start_CU2:
        .long .debug_str_offsets_end_CU2-.debug_str_offsets_base_CU2
        .short 5    # DWARF version
        .short 0    # Padding
.debug_str_offsets_base_CU2:
        .long str_producer-.debug_str.dwo
        .long str_CU2-.debug_str.dwo
        .long str_CU2_dir-.debug_str.dwo
.debug_str_offsets_end_CU2:

# TU2's contribution (from object file 2)
.debug_str_offsets_start_TU2:
        .long .debug_str_offsets_end_TU2-.debug_str_offsets_base_TU2
        .short 5    # DWARF version
        .short 0    # Padding
.debug_str_offsets_base_TU2:
        .long str_TU2-.debug_str.dwo
        .long str_TU2_type-.debug_str.dwo
.debug_str_offsets_end_TU2:


# Abbrevs are shared for all compile and type units
        .section .debug_abbrev.dwo,"",@progbits
        .byte 0x01  # Abbrev code
        .byte 0x11  # DW_TAG_compile_unit
        .byte 0x00  # DW_CHILDREN_no
        .byte 0x25  # DW_AT_producer
        .byte 0x1a  # DW_FORM_strx
        .byte 0x03  # DW_AT_name
        .byte 0x1a  # DW_FORM_strx
        .byte 0x72  # DW_AT_str_offsets_base
        .byte 0x17  # DW_FORM_sec_offset
        .byte 0x03  # DW_AT_name
        .byte 0x1a  # DW_FORM_strx
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x02  # Abbrev code
        .byte 0x41  # DW_TAG_type_unit
        .byte 0x01  # DW_CHILDREN_yes
        .byte 0x03  # DW_AT_name
        .byte 0x1a  # DW_FORM_strx
        .byte 0x72  # DW_AT_str_offsets_base
        .byte 0x17  # DW_FORM_sec_offset
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x03  # Abbrev code
        .byte 0x13  # DW_TAG_structure_type
        .byte 0x00  # DW_CHILDREN_no (no members)
        .byte 0x03  # DW_AT_name
        .byte 0x1a  # DW_FORM_strx
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x00  # EOM(3)
abbrev_end:

        .section .debug_info.dwo,"",@progbits

# DWARF v5 CU header.
CU1_5_start:
        .long  CU1_5_end-CU1_5_version  # Length of Unit
CU1_5_version:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev.dwo # Offset Into Abbrev. Section
# The compile-unit DIE, which has a DW_AT_producer, DW_AT_name,
# DW_AT_str_offsets and DW_AT_compdir.
        .byte 1                # Abbreviation code
        .byte 0                # The index of the producer string
        .byte 1                # The index of the CU name string
# The DW_AT_str_offsets_base attribute for CU1 contains the offset of CU1's
# contribution relative to the start of object file 1's portion of the
# .debug_str_offsets section.
        .long .debug_str_offsets_base_CU1-.debug_str_offsets_object_file1
        .byte 2                # The index of the comp dir string
        .byte 0 # NULL
CU1_5_end:

CU2_5_start:
        .long  CU2_5_end-CU2_5_version  # Length of Unit
CU2_5_version:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev.dwo # Offset Into Abbrev. Section
# The compile-unit DIE, which has a DW_AT_producer, DW_AT_name,
# DW_AT_str_offsets and DW_AT_compdir.
        .byte 1                # Abbreviation code
        .byte 0                # The index of the producer string
        .byte 1                # The index of the CU name string
# The DW_AT_str_offsets_base attribute for CU2 contains the offset of CU2's
# contribution relative to the start of object file 2's portion of the
# .debug_str_offsets section.
        .long .debug_str_offsets_base_CU2-.debug_str_offsets_object_file2
        .byte 2                # The index of the comp dir string
        .byte 0 # NULL
CU2_5_end:

        .section .debug_types.dwo,"",@progbits
# DWARF v5 Type unit header.
TU1_5_start:
        .long  TU1_5_end-TU1_5_version  # Length of Unit
TU1_5_version:
        .short 5               # DWARF version number
        .byte 2                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev.dwo    # Offset Into Abbrev. Section
        .quad 0x0011223344556677 # Type Signature
        .long TU1_5_type-TU1_5_start # Type offset
# The type-unit DIE, which has a name.
        .byte 2                # Abbreviation code
        .byte 0                # Index of the unit type name string
# The DW_AT_str_offsets_base attribute for TU1 contains the offset of TU1's
# contribution relative to the start of object file 1's portion of the
# .debug_str_offsets section.
        .long .debug_str_offsets_base_TU1-.debug_str_offsets_object_file1
# The type DIE, which has a name.
TU1_5_type:
        .byte 3                # Abbreviation code
        .byte 1                # Index of the type name string
        .byte 0 # NULL
        .byte 0 # NULL
TU1_5_end:

TU2_5_start:
        .long  TU2_5_end-TU2_5_version  # Length of Unit
TU2_5_version:
        .short 5               # DWARF version number
        .byte 2                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev.dwo    # Offset Into Abbrev. Section
        .quad 0x00aabbccddeeff99 # Type Signature
        .long TU2_5_type-TU2_5_start # Type offset
# The type-unit DIE, which has a name.
        .byte 2                # Abbreviation code
        .byte 0                # Index of the unit type name string
# The DW_AT_str_offsets_base attribute for TU2 contains the offset of TU2's
# contribution relative to the start of object file 2's portion of the
# .debug_str_offsets section.
        .long .debug_str_offsets_base_TU2-.debug_str_offsets_object_file2
# The type DIE, which has a name.
TU2_5_type:
        .byte 3                # Abbreviation code
        .byte 1                # Index of the type name string
        .byte 0 # NULL
        .byte 0 # NULL
TU2_5_end:

        .section .debug_cu_index,"",@progbits
        # The index header
        .long 2                # Version 
        .long 3                # Columns of contribution matrix
        .long 2                # number of units
        .long 2                # number of hash buckets in table

        # The signatures for both CUs.
        .quad 0xddeeaaddbbaabbee # signature 1
        .quad 0xff00ffeeffaaff00 # signature 2
        # The indexes for both CUs.
        .long 1                # index 1
        .long 2                # index 2
        # The sections to which both CUs contribute.
        .long 1                # DW_SECT_INFO
        .long 3                # DW_SECT_ABBREV
        .long 6                # DW_SECT_STR_OFFSETS

        # The starting offsets of both CU's contributions to info,
        # abbrev and string offsets table.
        .long CU1_5_start-.debug_info.dwo                   
        .long 0
        .long .debug_str_offsets_object_file1-.debug_str_offsets.dwo
        .long CU2_5_start-.debug_info.dwo
        .long 0
        .long .debug_str_offsets_object_file2-.debug_str_offsets.dwo

        # The lengths of both CU's contributions to info, abbrev and
        # string offsets table.
        .long CU1_5_end-CU1_5_start
        .long abbrev_end-.debug_abbrev.dwo
        .long .debug_str_offsets_end_CU1-.debug_str_offsets_start_CU1
        .long CU2_5_end-CU2_5_start
        .long abbrev_end-.debug_abbrev.dwo
        .long .debug_str_offsets_end_CU2-.debug_str_offsets_start_CU2

        .section .debug_tu_index,"",@progbits
        # The index header
        .long 2                # Version 
        .long 3                # Columns of contribution matrix
        .long 2                # number of units
        .long 2                # number of hash buckets in table

        # The signatures for both TUs.
        .quad 0xeeaaddbbaabbeedd # signature 1
        .quad 0x00ffeeffaaff00ff # signature 2
        # The indexes for both TUs.
        .long 1                # index 1
        .long 2                # index 2
        # The sections to which both TUs contribute.
        .long 2                # DW_SECT_TYPES
        .long 3                # DW_SECT_ABBREV
        .long 6                # DW_SECT_STR_OFFSETS

        # The starting offsets of both TU's contributions to info,
        # abbrev and string offsets table.
        .long TU1_5_start-.debug_types.dwo
        .long 0
        .long .debug_str_offsets_object_file1-.debug_str_offsets.dwo
        .long TU2_5_start-.debug_types.dwo
        .long 0
        .long .debug_str_offsets_object_file2-.debug_str_offsets.dwo

        # The lengths of both TU's contributions to info, abbrev and
        # string offsets table.
        .long TU1_5_end-TU1_5_start
        .long abbrev_end-.debug_abbrev.dwo
        .long .debug_str_offsets_end_TU1-.debug_str_offsets_start_TU1
        .long TU2_5_end-TU2_5_start
        .long abbrev_end-.debug_abbrev.dwo
        .long .debug_str_offsets_end_TU2-.debug_str_offsets_start_TU2
