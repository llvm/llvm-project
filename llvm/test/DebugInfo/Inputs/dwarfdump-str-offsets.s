# Test object to verify dwarfdump handles v5 string offset tables.
# We have 2 v5 CUs, a v5 TU, and a split v5 CU and TU.
#
# To generate the test object:
# llvm-mc -triple x86_64-unknown-linux dwarfdump-str-offsets.s -filetype=obj \
#         -o dwarfdump-str-offsets.x86_64.o

        .section .debug_str,"MS",@progbits,1
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
str_TU:
        .asciz "Type_Unit"
str_TU_type:
        .asciz "MyStruct"
str_Subprogram:
        .asciz "MyFunc"
str_Variable1:
        .asciz "MyVar1"
str_Variable2:
        .asciz "MyVar2"
str_Variable3:
        .asciz "MyVar3"

# Every unit contributes to the string_offsets table.
        .section .debug_str_offsets,"",@progbits
# CU1's contribution
        .long .debug_str_offsets_segment0_end-.debug_str_offsets_base0
        .short 5    # DWARF version
        .short 0    # Padding
.debug_str_offsets_base0:
        .long str_producer
        .long str_CU1
        .long str_CU1_dir
        .long str_Subprogram
        .long str_Variable1
        .long str_Variable2
        .long str_Variable3
.debug_str_offsets_segment0_end:
# CU2's contribution
        .long .debug_str_offsets_segment1_end-.debug_str_offsets_base1
        .short 5    # DWARF version
        .short 0    # Padding
.debug_str_offsets_base1:
        .long str_producer
        .long str_CU2
        .long str_CU2_dir
.debug_str_offsets_segment1_end:
# The TU's contribution
        .long .debug_str_offsets_segment2_end-.debug_str_offsets_base2
        .short 5    # DWARF version
        .short 0    # Padding
.debug_str_offsets_base2:
        .long str_TU
        .long str_TU_type
.debug_str_offsets_segment2_end:

        .section .debug_str.dwo,"MS",@progbits,1
dwo_str_CU_5_producer:
        .asciz "Handmade split DWARF producer"
dwo_str_CU_5_name:
        .asciz "V5_split_compile_unit"
dwo_str_CU_5_comp_dir:
        .asciz "/home/test/splitCU"
dwo_str_TU_5:
        .asciz "V5_split_type_unit"
dwo_str_TU_5_type:
        .asciz "V5_split_Mystruct"

        .section .debug_str_offsets.dwo,"",@progbits
# The split CU's contribution
        .long .debug_dwo_str_offsets_segment0_end-.debug_dwo_str_offsets_base0
        .short 5    # DWARF version
        .short 0    # Padding
.debug_dwo_str_offsets_base0:
        .long dwo_str_CU_5_producer-.debug_str.dwo
        .long dwo_str_CU_5_name-.debug_str.dwo
        .long dwo_str_CU_5_comp_dir-.debug_str.dwo
.debug_dwo_str_offsets_segment0_end:
# The split TU's contribution
        .long .debug_dwo_str_offsets_segment1_end-.debug_dwo_str_offsets_base1
        .short 5    # DWARF version
        .short 0    # Padding
.debug_dwo_str_offsets_base1:
        .long dwo_str_TU_5-.debug_str.dwo
        .long dwo_str_TU_5_type-.debug_str.dwo
.debug_dwo_str_offsets_segment1_end:

# All CUs/TUs use the same abbrev section for simplicity.
        .section .debug_abbrev,"",@progbits
        .byte 0x01  # Abbrev code
        .byte 0x11  # DW_TAG_compile_unit
        .byte 0x01  # DW_CHILDREN_yes
        .byte 0x25  # DW_AT_producer
        .byte 0x1a  # DW_FORM_strx
        .byte 0x03  # DW_AT_name
        .byte 0x1a  # DW_FORM_strx
        .byte 0x72  # DW_AT_str_offsets_base
        .byte 0x17  # DW_FORM_sec_offset
        .byte 0x1b  # DW_AT_comp_dir
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
        .byte 0x04  # Abbrev code
        .byte 0x2e  # DW_TAG_subprogram
        .byte 0x01  # DW_CHILDREN_yes
        .byte 0x03  # DW_AT_name
        .byte 0x25  # DW_FORM_strx1
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x05  # Abbrev code
        .byte 0x34  # DW_TAG_variable
        .byte 0x00  # DW_CHILDREN_no
        .byte 0x03  # DW_AT_name
        .byte 0x26  # DW_FORM_strx2
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x06  # Abbrev code
        .byte 0x34  # DW_TAG_variable
        .byte 0x00  # DW_CHILDREN_no
        .byte 0x03  # DW_AT_name
        .byte 0x27  # DW_FORM_strx3
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x07  # Abbrev code
        .byte 0x34  # DW_TAG_variable
        .byte 0x00  # DW_CHILDREN_no
        .byte 0x03  # DW_AT_name
        .byte 0x28  # DW_FORM_strx4
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x00  # EOM(3)

# And a .dwo copy of a subset for the .dwo sections.
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
        .byte 0x1b  # DW_AT_comp_dir
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
        
        .section .debug_info,"",@progbits

# DWARF v5 CU header.
        .long  CU1_5_end-CU1_5_version  # Length of Unit
CU1_5_version:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev    # Offset Into Abbrev. Section
# The compile-unit DIE, which has a DW_AT_producer, DW_AT_name, 
# DW_AT_str_offsets and DW_AT_compdir.
        .byte 1                # Abbreviation code
        .byte 0                # The index of the producer string
        .byte 1                # The index of the CU name string
        .long .debug_str_offsets_base0
        .byte 2                # The index of the comp dir string
# A subprogram DIE with DW_AT_name, using DW_FORM_strx1.
        .byte 4                # Abbreviation code
        .byte 3                # Subprogram name string (DW_FORM_strx1)
# A variable DIE with DW_AT_name, using DW_FORM_strx2.
        .byte 5                # Abbreviation code
        .short 0x0004          # Subprogram name string (DW_FORM_strx2)
# A variable DIE with DW_AT_name, using DW_FORM_strx3.
        .byte 6                # Abbreviation code
        .byte 5                # Subprogram name string (DW_FORM_strx3)
        .short 0               # Subprogram name string (DW_FORM_strx3)
# A variable DIE with DW_AT_name, using DW_FORM_strx4.
        .byte 7                # Abbreviation code
        .quad 0x00000006       # Subprogram name string (DW_FORM_strx4)
        .byte 0 # NULL
        .byte 0 # NULL
        .byte 0 # NULL
CU1_5_end:

# DWARF v5 CU header
        .long  CU2_5_end-CU2_5_version  # Length of Unit
CU2_5_version:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev    # Offset Into Abbrev. Section
# The compile-unit DIE, which has a DW_AT_producer, DW_AT_name, 
# DW_AT_str_offsets and DW_AT_compdir.
        .byte 1                # Abbreviation code
        .byte 0                # The index of the producer string
        .byte 1                # The index of the CU name string
        .long .debug_str_offsets_base1
        .byte 2                # The index of the comp dir string
        .byte 0 # NULL
CU2_5_end:

        .section .debug_types,"",@progbits
# DWARF v5 Type unit header.
TU_5_start:
        .long  TU_5_end-TU_5_version  # Length of Unit
TU_5_version:
        .short 5               # DWARF version number
        .byte 2                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev    # Offset Into Abbrev. Section
        .quad 0x0011223344556677 # Type Signature
        .long TU_5_type-TU_5_start # Type offset
# The type-unit DIE, which has a name.
        .byte 2                # Abbreviation code
        .byte 0                # Index of the unit type name string
        .long .debug_str_offsets_base2  # offset into the str_offsets section
# The type DIE, which has a name.
TU_5_type:
        .byte 3                # Abbreviation code
        .byte 1                # Index of the type name string
        .byte 0 # NULL
        .byte 0 # NULL
TU_5_end:
        
        .section .debug_info.dwo,"",@progbits

# DWARF v5 split CU header.
        .long  CU_split_5_end-CU_split_5_version  # Length of Unit
CU_split_5_version:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev.dwo  # Offset Into Abbrev Section
# The compile-unit DIE, which has a DW_AT_producer, DW_AT_name, 
# DW_AT_str_offsets and DW_AT_compdir.
        .byte 1                # Abbreviation code
        .byte 0                # The index of the producer string
        .byte 1                # The index of the CU name string
        .long .debug_dwo_str_offsets_base0-.debug_str_offsets.dwo
        .byte 2                # The index of the comp dir string
        .byte 0 # NULL
CU_split_5_end:
        
        .section .debug_types.dwo,"",@progbits

# DWARF v5 split type unit header.
TU_split_5_start:
        .long  TU_split_5_end-TU_split_5_version  # Length of Unit
TU_split_5_version:
        .short 5               # DWARF version number
        .byte 6                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev.dwo  # Offset Into Abbrev Section
        .quad 0x8899aabbccddeeff # Type Signature
        .long TU_split_5_type-TU_split_5_start  # Type offset
# The type-unit DIE, which has a name.
        .byte 2                # Abbreviation code
        .byte 0                # The index of the type unit name string
        .long .debug_dwo_str_offsets_base1-.debug_str_offsets.dwo 
# The type DIE, which has a name.
TU_split_5_type:
        .byte 3                # Abbreviation code
        .byte 1                # The index of the type name string
        .byte 0 # NULL
        .byte 0 # NULL
TU_split_5_end:
