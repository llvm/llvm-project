# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.o
# RUN: llvm-dwarfdump -v %t.o 2> %t.err | FileCheck --check-prefixes=COMMON,SPLIT,OFFSETS %s
# RUN: llvm-dwarfdump -verify %t.o | FileCheck --check-prefix=VERIFY %s
# RUN: llvm-dwarfdump -debug-str-offsets %t.o | FileCheck --check-prefix=OFFSETS %s
# 
# Check that we don't report an error on a non-existent range list table.
# RUN: FileCheck -allow-empty --check-prefix ERR %s < %t.err

# Test object to verify dwarfdump handles v5 string offset tables.
# We have 2 v5 CUs, a v5 TU, and a split v5 CU and TU.
#

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
        .long .debug_str_offsets_segment0_end-.debug_str_offsets_base0+4
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
# A 4-byte gap.
        .long 0
# CU2's contribution in DWARF64 format
        .long 0xffffffff
        .quad .debug_str_offsets_segment1_end-.debug_str_offsets_base1+4
        .short 5    # DWARF version
        .short 0    # Padding
.debug_str_offsets_base1:
        .quad str_producer
        .quad str_CU2
        .quad str_CU2_dir
.debug_str_offsets_segment1_end:
# The TU's contribution
        .long .debug_str_offsets_segment2_end-.debug_str_offsets_base2+4
        .short 5    # DWARF version
        .short 0    # Padding
.debug_str_offsets_base2:
        .long str_TU
        .long str_TU_type
.debug_str_offsets_segment2_end:

        .section .debug_str.dwo,"MSe",@progbits,1
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

        .section .debug_str_offsets.dwo,"e",@progbits
# One contribution only in a .dwo file
        .long .debug_dwo_str_offsets_segment0_end-.debug_dwo_str_offsets_base0+4
        .short 5    # DWARF version
        .short 0    # Padding
.debug_dwo_str_offsets_base0:
        .long dwo_str_CU_5_producer-.debug_str.dwo
        .long dwo_str_CU_5_name-.debug_str.dwo
        .long dwo_str_CU_5_comp_dir-.debug_str.dwo
        .long dwo_str_TU_5-.debug_str.dwo
        .long dwo_str_TU_5_type-.debug_str.dwo
.debug_dwo_str_offsets_segment0_end:

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
        .byte 0x49  # DW_AT_type
        .byte 0x13  # DW_FORM_ref4
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x06  # Abbrev code
        .byte 0x34  # DW_TAG_variable
        .byte 0x00  # DW_CHILDREN_no
        .byte 0x03  # DW_AT_name
        .byte 0x27  # DW_FORM_strx3
        .byte 0x49  # DW_AT_type
        .byte 0x13  # DW_FORM_ref4
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x07  # Abbrev code
        .byte 0x34  # DW_TAG_variable
        .byte 0x00  # DW_CHILDREN_no
        .byte 0x03  # DW_AT_name
        .byte 0x28  # DW_FORM_strx4
        .byte 0x49  # DW_AT_type
        .byte 0x13  # DW_FORM_ref4
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x08  # Abbrev code
        .byte 0x24  # DW_TAG_base_type
        .byte 0x00  # DW_CHILDREN_no
        .byte 0x3e  # DW_AT_encoding
        .byte 0x0b  # DW_FORM_data1
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x09  # Abbrev code
        .byte 0x11  # DW_TAG_compile_unit
        .byte 0x00  # DW_CHILDREN_no
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x00  # EOM(3)

# And a .dwo copy of a subset for the .dwo sections.
        .section .debug_abbrev.dwo,"e",@progbits
        .byte 0x01  # Abbrev code
        .byte 0x11  # DW_TAG_compile_unit
        .byte 0x00  # DW_CHILDREN_no
        .byte 0x25  # DW_AT_producer
        .byte 0x1a  # DW_FORM_strx
        .byte 0x03  # DW_AT_name
        .byte 0x1a  # DW_FORM_strx
        .byte 0x1b  # DW_AT_comp_dir
        .byte 0x1a  # DW_FORM_strx
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x02  # Abbrev code
        .byte 0x41  # DW_TAG_type_unit
        .byte 0x01  # DW_CHILDREN_yes
        .byte 0x03  # DW_AT_name
        .byte 0x1a  # DW_FORM_strx
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
# A variable DIE with DW_AT_name, using DW_FORM_strx2, and DW_AT_type.
        .byte 5                # Abbreviation code
        .short 0x0004          # Subprogram name string (DW_FORM_strx2)
        .long TypeDie-.debug_info
# A variable DIE with DW_AT_name, using DW_FORM_strx3, and DW_AT_type.
        .byte 6                # Abbreviation code
        .byte 5                # Subprogram name string (DW_FORM_strx3)
        .short 0               # Subprogram name string (DW_FORM_strx3)
        .long TypeDie-.debug_info
# A variable DIE with DW_AT_name, using DW_FORM_strx4, and DW_AT_type.
        .byte 7                # Abbreviation code
        .long 6                # Subprogram name string (DW_FORM_strx4)
        .long TypeDie-.debug_info
        .byte 0 # NULL
# A base type DIE with DW_AT_encoding.
TypeDie:
        .byte 8                # Abbreviation code
        .byte 5                # DW_ATE_signed
        .byte 0 # NULL
        .byte 0 # NULL
CU1_5_end:

# DWARF v5 CU header
        .long 0xffffffff
        .quad CU2_5_end-CU2_5_version  # Length of Unit
CU2_5_version:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .quad .debug_abbrev    # Offset Into Abbrev. Section
# The compile-unit DIE, which has a DW_AT_producer, DW_AT_name, 
# DW_AT_str_offsets and DW_AT_compdir.
        .byte 1                # Abbreviation code
        .byte 0                # The index of the producer string
        .byte 1                # The index of the CU name string
        .quad .debug_str_offsets_base1
        .byte 2                # The index of the comp dir string
        .byte 0 # NULL
CU2_5_end:
# DWARF v5 CU without str_offsets_base - this shouldn't produce an error/nor
# prevent other str_offsets contributions from being dumped.
	.long CU3_5_end-CU3_5_version  # Length of Unit
CU3_5_version:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev    # Offset Into Abbrev. Section
# The compile-unit DIE with no attributes.
        .byte 9                # Abbreviation code
CU3_5_end:






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
        
        .section .debug_info.dwo,"e",@progbits

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
        .byte 2                # The index of the comp dir string
        .byte 0 # NULL
CU_split_5_end:
        
        .section .debug_types.dwo,"e",@progbits

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
        .byte 3                # The index of the type unit name string
# The type DIE, which has a name.
TU_split_5_type:
        .byte 3                # Abbreviation code
        .byte 4                # The index of the type name string
        .byte 0 # NULL
        .byte 0 # NULL
TU_split_5_end:

# We are using a hand-constructed object file and are interest in the correct
# diplay of the DW_str_offsetsbase attribute, the correct display of strings
# and the dump of the .debug_str_offsets[.dwo] table.

# Abbreviation for DW_AT_str_offsets_base
# COMMON:      .debug_abbrev contents:
# COMMON-NOT:  contents:
# COMMON:      DW_TAG_compile_unit
# COMMON-NOT:  DW_TAG
# COMMON:      DW_AT_str_offsets_base DW_FORM_sec_offset
# 
# Verify that strings are displayed correctly as indexed strings
# COMMON:      .debug_info contents:
# COMMON-NOT:  contents:     
# COMMON:      DW_TAG_compile_unit
# COMMON-NEXT: DW_AT_producer [DW_FORM_strx] (indexed (00000000) string = "Handmade DWARF producer")
# COMMON-NEXT: DW_AT_name [DW_FORM_strx] (indexed (00000001) string = "Compile_Unit_1")
# COMMON-NEXT: DW_AT_str_offsets_base [DW_FORM_sec_offset] (0x00000008)
# COMMON-NEXT: DW_AT_comp_dir [DW_FORM_strx] (indexed (00000002) string = "/home/test/CU1")
# COMMON-NOT:  NULL
# COMMON:      DW_TAG_subprogram
# COMMON-NEXT: DW_AT_name [DW_FORM_strx1] (indexed (00000003) string = "MyFunc")
# COMMON-NOT:  NULL
# COMMON:      DW_TAG_variable
# COMMON-NEXT: DW_AT_name [DW_FORM_strx2] (indexed (00000004) string = "MyVar1")
# COMMON-NOT:  NULL
# COMMON:      DW_TAG_variable
# COMMON-NEXT: DW_AT_name [DW_FORM_strx3] (indexed (00000005) string = "MyVar2")
# COMMON-NOT:  NULL
# COMMON:      DW_TAG_variable
# COMMON-NEXT: DW_AT_name [DW_FORM_strx4] (indexed (00000006) string = "MyVar3")
# 
# Second compile unit (b.cpp)
# COMMON:      DW_TAG_compile_unit
# COMMON-NEXT: DW_AT_producer [DW_FORM_strx] (indexed (00000000) string = "Handmade DWARF producer")
# COMMON-NEXT: DW_AT_name [DW_FORM_strx] (indexed (00000001) string = "Compile_Unit_2")
# COMMON-NEXT: DW_AT_str_offsets_base [DW_FORM_sec_offset] (0x0000000000000038)
# COMMON-NEXT: DW_AT_comp_dir [DW_FORM_strx] (indexed (00000002) string = "/home/test/CU2")
# 
# The split CU
# SPLIT:       .debug_info.dwo contents:
# SPLIT-NOT:   contents:
# SPLIT:       DW_TAG_compile_unit
# SPLIT-NEXT:  DW_AT_producer [DW_FORM_strx] (indexed (00000000) string = "Handmade split DWARF producer")
# SPLIT-NEXT:  DW_AT_name [DW_FORM_strx] (indexed (00000001) string = "V5_split_compile_unit")
# SPLIT-NEXT:  DW_AT_comp_dir [DW_FORM_strx] (indexed (00000002) string = "/home/test/splitCU")
# 
# The type unit
# COMMON:      .debug_types contents:
# COMMON:      DW_TAG_type_unit
# COMMON-NEXT: DW_AT_name [DW_FORM_strx] (indexed (00000000) string = "Type_Unit")
# COMMON:      DW_TAG_structure_type
# COMMON-NEXT: DW_AT_name [DW_FORM_strx] (indexed (00000001) string = "MyStruct")
# 
# The split type unit
# SPLIT:       .debug_types.dwo contents:
# SPLIT:       DW_TAG_type_unit
# SPLIT-NEXT:  DW_AT_name [DW_FORM_strx] (indexed (00000003) string = "V5_split_type_unit")
# SPLIT:       DW_TAG_structure_type
# SPLIT-NEXT:  DW_AT_name [DW_FORM_strx] (indexed (00000004) string = "V5_split_Mystruct")
# 
# The .debug_str_offsets section
# OFFSETS:      .debug_str_offsets contents:
# OFFSETS-NEXT: 0x00000000: Contribution size = 32, Format = DWARF32, Version = 5
# OFFSETS-NEXT: 0x00000008: 00000000 "Handmade DWARF producer"
# OFFSETS-NEXT: 0x0000000c: 00000018 "Compile_Unit_1"
# OFFSETS-NEXT: 0x00000010: 00000027 "/home/test/CU1"
# OFFSETS-NEXT: 0x00000014: 00000067 "MyFunc"
# OFFSETS-NEXT: 0x00000018: 0000006e "MyVar1"
# OFFSETS-NEXT: 0x0000001c: 00000075 "MyVar2"
# OFFSETS-NEXT: 0x00000020: 0000007c "MyVar3"
# OFFSETS-NEXT: Gap, length = 4
# OFFSETS-NEXT: 0x00000028: Contribution size = 28, Format = DWARF64, Version = 5
# OFFSETS-NEXT: 0x00000038: 0000000000000000 "Handmade DWARF producer"
# OFFSETS-NEXT: 0x00000040: 0000000000000036 "Compile_Unit_2"
# OFFSETS-NEXT: 0x00000048: 0000000000000045 "/home/test/CU2"
# OFFSETS-NEXT: 0x00000050: Contribution size = 12, Format = DWARF32, Version = 5
# OFFSETS-NEXT: 0x00000058: 00000054 "Type_Unit"
# OFFSETS-NEXT: 0x0000005c: 0000005e "MyStruct"
# 
# OFFSETS:       .debug_str_offsets.dwo contents:
# OFFSETS-NEXT:  0x00000000: Contribution size = 24, Format = DWARF32, Version = 5
# OFFSETS-NEXT:  0x00000008: 00000000 "Handmade split DWARF producer"
# OFFSETS-NEXT:  0x0000000c: 0000001e "V5_split_compile_unit"
# OFFSETS-NEXT:  0x00000010: 00000034 "/home/test/splitCU"
# OFFSETS-NEXT:  0x00000014: 00000047 "V5_split_type_unit"
# OFFSETS-NEXT:  0x00000018: 0000005a "V5_split_Mystruct"

# VERIFY: No errors.

# ERR-NOT: parsing a range list table:
