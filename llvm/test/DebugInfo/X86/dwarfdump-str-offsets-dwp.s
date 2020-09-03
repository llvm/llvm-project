# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.o
# RUN: llvm-dwarfdump -v %t.o | FileCheck %s

# Test object to verify that dwarfdump handles dwp files with DWARF v5 string
# offset tables. We have 3 CUs and 2 TUs, where it is assumed that 
# CU1 and TU1 came from one object file, CU2 and TU2 from a second object
# file, and CU3 from a third object file that was compiled with 
# -gdwarf-4.
#
        .section .debug_str.dwo,"MSe",@progbits,1
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
str_CU3:
        .asciz "Compile_Unit_3"
str_CU3_dir:
        .asciz "/home/test/CU3"

        .section .debug_str_offsets.dwo,"e",@progbits
# Object files 1's portion of the .debug_str_offsets.dwo section.
# CU1 and TU1 share a contribution to the string offsets table.
.debug_str_offsets_object_file1_start:
        .long .debug_str_offsets_object_file1_end-.debug_str_offsets_base_1
        .short 5    # DWARF version
        .short 0    # Padding
.debug_str_offsets_base_1:
        .long str_producer-.debug_str.dwo
        .long str_CU1-.debug_str.dwo
        .long str_CU1_dir-.debug_str.dwo
        .long str_TU1-.debug_str.dwo
        .long str_TU1_type-.debug_str.dwo
.debug_str_offsets_object_file1_end:

# Object files 2's portion of the .debug_str_offsets.dwo section.
# CU2 and TU2 share a contribution to the string offsets table.
.debug_str_offsets_object_file2_start:
        .long .debug_str_offsets_object_file2_end-.debug_str_offsets_base_2
        .short 5    # DWARF version
        .short 0    # Padding
.debug_str_offsets_base_2:
        .long str_producer-.debug_str.dwo
        .long str_CU2-.debug_str.dwo
        .long str_CU2_dir-.debug_str.dwo
        .long str_TU2-.debug_str.dwo
        .long str_TU2_type-.debug_str.dwo
.debug_str_offsets_object_file2_end:

# Object files 3's portion of the .debug_str_offsets.dwo section.
# This file is assumed to have been compiled with -gdwarf-4 and
# therefore contains a version 4 CU and a GNU format contribution
# to the .debug_str_offsets section.
.debug_str_offsets_object_file3_start:
.debug_str_offsets_base_3:
        .long str_producer-.debug_str.dwo
        .long str_CU3-.debug_str.dwo
        .long str_CU3_dir-.debug_str.dwo
.debug_str_offsets_object_file3_end:

# Abbrevs are shared for all compile and type units
        .section .debug_abbrev.dwo,"e",@progbits
        .byte 0x01  # Abbrev code
        .byte 0x11  # DW_TAG_compile_unit
        .byte 0x00  # DW_CHILDREN_no
        .byte 0x25  # DW_AT_producer
        .byte 0x1a  # DW_FORM_strx
        .byte 0x03  # DW_AT_name
        .byte 0x1a  # DW_FORM_strx
        .byte 0x03  # DW_AT_name
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
        .byte 0x04  # Abbrev code
        .byte 0x11  # DW_TAG_compile_unit
        .byte 0x00  # DW_CHILDREN_no
        .byte 0x25  # DW_AT_producer
        .short 0x3e82  # DW_FORM_GNU_str_index
        .byte 0x03  # DW_AT_name
        .short 0x3e82  # DW_FORM_GNU_str_index
        .byte 0x03  # DW_AT_name
        .short 0x3e82  # DW_FORM_GNU_str_index
        .byte 0x00  # EOM(1)
        .byte 0x00  # EOM(2)
        .byte 0x00  # EOM(3)
abbrev_end:

        .section .debug_info.dwo,"e",@progbits

# DWARF v5 CU header.
CU1_5_start:
        .long  CU1_5_end-CU1_5_version  # Length of Unit
CU1_5_version:
        .short 5               # DWARF version number
        .byte 1                # DWARF Unit Type
        .byte 8                # Address Size (in bytes)
        .long .debug_abbrev.dwo # Offset Into Abbrev. Section
# The compile-unit DIE, which has a DW_AT_producer, DW_AT_name
# and DW_AT_compdir.
        .byte 1                # Abbreviation code
        .byte 0                # The index of the producer string
        .byte 1                # The index of the CU name string
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
# The compile-unit DIE, which has a DW_AT_producer, DW_AT_name
# and DW_AT_compdir.
        .byte 1                # Abbreviation code
        .byte 0                # The index of the producer string
        .byte 1                # The index of the CU name string
        .byte 2                # The index of the comp dir string
        .byte 0 # NULL
CU2_5_end:

CU3_4_start:
        .long  CU3_4_end-CU3_4_version  # Length of Unit
CU3_4_version:
        .short 4               # DWARF version number
        .long .debug_abbrev.dwo # Offset Into Abbrev. Section
        .byte 8                # Address Size (in bytes)
# The compile-unit DIE, which has a DW_AT_producer, DW_AT_name
# and DW_AT_compdir.
        .byte 4                # Abbreviation code
        .byte 0                # The index of the producer string
        .byte 1                # The index of the CU name string
        .byte 2                # The index of the comp dir string
        .byte 0 # NULL
CU3_4_end:

        .section .debug_types.dwo,"e",@progbits
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
        .byte 3                # Index of the unit type name string
# The type DIE, which has a name.
TU1_5_type:
        .byte 3                # Abbreviation code
        .byte 4                # Index of the type name string
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
        .byte 3                # Index of the unit type name string
# The type DIE, which has a name.
TU2_5_type:
        .byte 3                # Abbreviation code
        .byte 4                # Index of the type name string
        .byte 0 # NULL
        .byte 0 # NULL
TU2_5_end:

        .section .debug_cu_index,"",@progbits
        # The index header
        .long 2                # Version 
        .long 3                # Columns of contribution matrix
        .long 3                # number of units
        .long 3                # number of hash buckets in table

        # The signatures for all CUs.
        .quad 0xddeeaaddbbaabbee # signature 1
        .quad 0xff00ffeeffaaff00 # signature 2
        .quad 0xf00df00df00df00d # signature 2
        # The indexes for both CUs.
        .long 1                # index 1
        .long 2                # index 2
        .long 3                # index 3
        # The sections to which all CUs contribute.
        .long 1                # DW_SECT_INFO
        .long 3                # DW_SECT_ABBREV
        .long 6                # DW_SECT_STR_OFFSETS

        # The starting offsets of all CU's contributions to info,
        # abbrev and string offsets table.
        .long CU1_5_start-.debug_info.dwo                   
        .long 0
        .long .debug_str_offsets_object_file1_start-.debug_str_offsets.dwo
        .long CU2_5_start-.debug_info.dwo
        .long 0
        .long .debug_str_offsets_object_file2_start-.debug_str_offsets.dwo
        .long CU3_4_start-.debug_info.dwo
        .long 0
        .long .debug_str_offsets_object_file3_start-.debug_str_offsets.dwo

        # The lengths of all CU's contributions to info, abbrev and
        # string offsets table.
        .long CU1_5_end-CU1_5_start
        .long abbrev_end-.debug_abbrev.dwo
        .long .debug_str_offsets_object_file1_end-.debug_str_offsets_object_file1_start
        .long CU2_5_end-CU2_5_start
        .long abbrev_end-.debug_abbrev.dwo
        .long .debug_str_offsets_object_file2_end-.debug_str_offsets_object_file2_start
        .long CU3_4_end-CU3_4_start
        .long abbrev_end-.debug_abbrev.dwo
        .long .debug_str_offsets_object_file3_end-.debug_str_offsets_object_file3_start

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
        .long .debug_str_offsets_object_file1_start-.debug_str_offsets.dwo
        .long TU2_5_start-.debug_types.dwo
        .long 0
        .long .debug_str_offsets_object_file2_start-.debug_str_offsets.dwo

        # The lengths of both TU's contributions to info, abbrev and
        # string offsets table.
        .long TU1_5_end-TU1_5_start
        .long abbrev_end-.debug_abbrev.dwo
        .long .debug_str_offsets_object_file1_end-.debug_str_offsets_object_file1_start
        .long TU2_5_end-TU2_5_start
        .long abbrev_end-.debug_abbrev.dwo
        .long .debug_str_offsets_object_file2_end-.debug_str_offsets_object_file2_start


# Verify that the correct strings from each unit are displayed and that the
# index for the .debug_str_offsets section has the right values.

# CHECK:      Compile Unit
# CHECK-NOT:  NULL
# CHECK:      DW_TAG_compile_unit
# CHECK-NEXT: DW_AT_producer [DW_FORM_strx] (indexed (00000000) string = "Handmade DWARF producer")
# CHECK-NEXT: DW_AT_name [DW_FORM_strx] (indexed (00000001) string = "Compile_Unit_1")
# CHECK-NEXT: DW_AT_name [DW_FORM_strx] (indexed (00000002) string = "/home/test/CU1")
# CHECK-NOT:  NULL

# CHECK:      Compile Unit
# CHECK-NOT:  NULL
# CHECK:      DW_TAG_compile_unit
# CHECK-NEXT: DW_AT_producer [DW_FORM_strx] (indexed (00000000) string = "Handmade DWARF producer")
# CHECK-NEXT: DW_AT_name [DW_FORM_strx] (indexed (00000001) string = "Compile_Unit_2")
# CHECK-NEXT: DW_AT_name [DW_FORM_strx] (indexed (00000002) string = "/home/test/CU2")
# 
# CHECK:      Type Unit
# CHECK-NOT:  NULL
# CHECK:      DW_TAG_type_unit
# CHECK-NEXT: DW_AT_name [DW_FORM_strx] (indexed (00000003) string = "Type_Unit_1")
# CHECK-NOT:  NULL
# CHECK:      DW_TAG_structure_type
# CHECK-NEXT: DW_AT_name [DW_FORM_strx] (indexed (00000004) string = "MyStruct_1")
#
# CHECK:      Type Unit
# CHECK-NOT:  NULL
# CHECK:      DW_TAG_type_unit
# CHECK-NEXT: DW_AT_name [DW_FORM_strx] (indexed (00000003) string = "Type_Unit_2")
# CHECK-NOT:  NULL
# CHECK:      DW_TAG_structure_type
# CHECK-NEXT: DW_AT_name [DW_FORM_strx] (indexed (00000004) string = "MyStruct_2")

# Verify the correct offets of the compile and type units contributions in the
# index tables.

# CHECK:      .debug_cu_index contents:
# CHECK-NOT:  contents:
# CHECK:      1 0xddeeaaddbbaabbee [{{0x[0-9a-f]*, 0x[0-9a-f]*}}) [{{0x[0-9a-f]*, 0x[0-9a-f]*}})
# CHECK-SAME: [0x00000000
# CHECK-NEXT: 2 0xff00ffeeffaaff00 [{{0x[0-9a-f]*, 0x[0-9a-f]*}}) [{{0x[0-9a-f]*, 0x[0-9a-f]*}})
# CHECK-SAME: [0x0000001c

# CHECK:      .debug_tu_index contents:
# CHECK-NOT:  contents:
# CHECK:      1 0xeeaaddbbaabbeedd [{{0x[0-9a-f]*, 0x[0-9a-f]*}}) [{{0x[0-9a-f]*, 0x[0-9a-f]*}})
# CHECK-SAME: [0x00000000
# CHECK-NEXT: 2 0x00ffeeffaaff00ff [{{0x[0-9a-f]*, 0x[0-9a-f]*}}) [{{0x[0-9a-f]*, 0x[0-9a-f]*}})
# CHECK:      [0x0000001c
