## The test checks that v5 compile units in package files read their
## location tables from .debug_loclists.dwo sections.
## See dwp-v2-loc.s for pre-v5 units.

# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o - | \
# RUN:   llvm-dwarfdump -debug-info -debug-loclists - | \
# RUN:   FileCheck %s

# CHECK:      .debug_info.dwo contents:
# CHECK:      DW_TAG_compile_unit
# CHECK:      DW_TAG_variable
# CHECK-NEXT:   DW_AT_name ("a")
# CHECK-NEXT:   DW_AT_location (0x{{[0-9a-f]+}}:
# CHECK-NEXT:     DW_LLE_startx_length (0x0000000000000001, 0x0000000000000010): DW_OP_reg5 RDI)
# CHECK:      DW_TAG_variable
# CHECK-NEXT:   DW_AT_name ("b")
# CHECK-NEXT:   DW_AT_location (indexed (0x1) loclist = 0x{{[0-9a-f]+}}:
# CHECK-NEXT:     DW_LLE_startx_length (0x0000000000000005, 0x0000000000000020): DW_OP_regx RDI)

# CHECK:      .debug_loclists.dwo contents:
# CHECK:      locations list header:
# CHECK:      locations list header:
# CHECK:      offsets:
# CHECK:      0x{{[0-9a-f]+}}:
# CHECK-NEXT:   DW_LLE_startx_length (0x0000000000000001, 0x0000000000000010): DW_OP_reg5 RDI
# CHECK:      0x{{[0-9a-f]+}}:
# CHECK-NEXT:   DW_LLE_startx_length (0x0000000000000005, 0x0000000000000020): DW_OP_regx RDI

.section .debug_abbrev.dwo, "e", @progbits
.LAbbrevBegin:
    .uleb128 1                          # Abbreviation Code
    .uleb128 17                         # DW_TAG_compile_unit
    .byte 1                             # DW_CHILDREN_yes
    .byte 0                             # EOM(1)
    .byte 0                             # EOM(2)
    .uleb128 2                          # Abbreviation Code
    .uleb128 52                         # DW_TAG_variable
    .byte 0                             # DW_CHILDREN_no
    .uleb128 3                          # DW_AT_name
    .uleb128 8                          # DW_FORM_string
    .uleb128 2                          # DW_AT_location
    .uleb128 23                         # DW_FORM_sec_offset
    .byte 0                             # EOM(1)
    .byte 0                             # EOM(2)
    .uleb128 3                          # Abbreviation Code
    .uleb128 52                         # DW_TAG_variable
    .byte 0                             # DW_CHILDREN_no
    .uleb128 3                          # DW_AT_name
    .uleb128 8                          # DW_FORM_string
    .uleb128 2                          # DW_AT_location
    .uleb128 34                         # DW_FORM_loclistx
    .byte 0                             # EOM(1)
    .byte 0                             # EOM(2)
    .byte 0                             # EOM(3)
.LAbbrevEnd:

    .section .debug_info.dwo, "e", @progbits
.LCUBegin:
    .long .LCUEnd-.LCUVersion           # Length of Unit
.LCUVersion:
    .short 5                            # Version
    .byte 5                             # DW_UT_split_compile
    .byte 8                             # Address size
    .long 0                             # Abbrev offset
    .quad 0x1100001122222222            # DWO id
    .uleb128 1                          # Abbrev [1] DW_TAG_compile_unit
    .uleb128 2                          # Abbrev [2] DW_TAG_variable
    .asciz "a"                          # DW_AT_name
    .long .LLL0-.LLLBegin               # DW_AT_location (DW_FORM_sec_offset)
    .uleb128 3                          # Abbrev [3] DW_TAG_variable
    .asciz "b"                          # DW_AT_name
    .uleb128 1                          # DW_AT_location (DW_FORM_loclistx)
    .byte 0                             # End Of Children Mark
.LCUEnd:

.section .debug_loclists.dwo, "e", @progbits
## Start the section with an unused table to check that the reading offset
## of the real table is correctly adjusted.
    .long .LLLDummyEnd-.LLLDummyVersion # Length of Unit
.LLLDummyVersion:
    .short 5                            # Version
    .byte 8                             # Address size
    .byte 0                             # Segment selector size
    .long 0                             # Offset entry count
    .byte 0                             # DW_LLE_end_of_list
.LLLDummyEnd:

.LLLBegin:
    .long .LLLEnd-.LLLVersion           # Length of Unit
.LLLVersion:
    .short 5                            # Version
    .byte 8                             # Address size
    .byte 0                             # Segment selector size
    .long 2                             # Offset entry count
.LLLBase:
    .long .LLL0-.LLLBase
    .long .LLL1-.LLLBase
.LLL0:
    .byte 3                             # DW_LLE_startx_length
    .uleb128 1                          # Index
    .uleb128 0x10                       # Length
    .uleb128 1                          # Loc expr size
    .byte 85                            # DW_OP_reg5
    .byte 0                             # DW_LLE_end_of_list
.LLL1:
    .byte 3                             # DW_LLE_startx_length
    .uleb128 5                          # Index
    .uleb128 0x20                       # Length
    .uleb128 2                          # Loc expr size
    .byte 144                           # DW_OP_regx
    .uleb128 5                          # RDI
    .byte 0                             # DW_LLE_end_of_list
.LLLEnd:

    .section .debug_cu_index, "", @progbits
## Header:
    .short 5                            # Version
    .space 2                            # Padding
    .long 3                             # Section count
    .long 1                             # Unit count
    .long 2                             # Slot count
## Hash Table of Signatures:
    .quad 0x1100001122222222
    .quad 0
## Parallel Table of Indexes:
    .long 1
    .long 0
## Table of Section Offsets:
## Row 0:
    .long 1                             # DW_SECT_INFO
    .long 3                             # DW_SECT_ABBREV
    .long 5                             # DW_SECT_LOCLISTS
## Row 1:
    .long 0                             # Offset in .debug_info.dwo
    .long 0                             # Offset in .debug_abbrev.dwo
    .long .LLLBegin-.debug_loclists.dwo # Offset in .debug_loclists.dwo
## Table of Section Sizes:
    .long .LCUEnd-.LCUBegin             # Size in .debug_info.dwo
    .long .LAbbrevEnd-.LAbbrevBegin     # Size in .debug_abbrev.dwo
    .long .LLLEnd-.LLLBegin             # Size in .debug_loclists.dwo
