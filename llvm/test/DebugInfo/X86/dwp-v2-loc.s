## The test checks that pre-v5 compile units in package files read their
## location tables from .debug_loc.dwo sections.
## See dwp-v5-loclists.s for v5 units.

# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o - | \
# RUN:   llvm-dwarfdump -debug-info -debug-loc - | \
# RUN:   FileCheck %s

# CHECK:      .debug_info.dwo contents:
# CHECK:      DW_TAG_compile_unit
# CHECK-NEXT:   DW_AT_GNU_dwo_id (0x1100001122222222)
# CHECK:      DW_TAG_variable
# CHECK-NEXT:   DW_AT_name ("a")
# CHECK-NEXT:   DW_AT_location (0x00000000:
# CHECK-NEXT:     DW_LLE_startx_length (0x0000000000000001, 0x0000000000000010): DW_OP_reg5 RDI)

# CHECK:      .debug_loc.dwo contents:
# CHECK:      0x00000013:
# CHECK-NEXT:   DW_LLE_startx_length (0x00000001, 0x00000010): DW_OP_reg5 RDI

.section .debug_abbrev.dwo, "e", @progbits
.LAbbrevBegin:
    .uleb128 1                      # Abbreviation Code
    .uleb128 17                     # DW_TAG_compile_unit
    .byte 1                         # DW_CHILDREN_yes
    .uleb128 0x2131                 # DW_AT_GNU_dwo_id
    .uleb128 7                      # DW_FORM_data8
    .byte 0                         # EOM(1)
    .byte 0                         # EOM(2)
    .uleb128 2                      # Abbreviation Code
    .uleb128 52                     # DW_TAG_variable
    .byte 0                         # DW_CHILDREN_no
    .uleb128 3                      # DW_AT_name
    .uleb128 8                      # DW_FORM_string
    .uleb128 2                      # DW_AT_location
    .uleb128 23                     # DW_FORM_sec_offset
    .byte 0                         # EOM(1)
    .byte 0                         # EOM(2)
    .byte 0                         # EOM(3)
.LAbbrevEnd:

    .section .debug_info.dwo, "e", @progbits
.LCUBegin:
    .long .LCUEnd-.LCUVersion       # Length of Unit
.LCUVersion:
    .short 4                        # Version
    .long 0                         # Abbrev offset
    .byte 8                         # Address size
    .uleb128 1                      # Abbrev [1] DW_TAG_compile_unit
    .quad 0x1100001122222222        # DW_AT_GNU_dwo_id
    .uleb128 2                      # Abbrev [2] DW_TAG_variable
    .asciz "a"                      # DW_AT_name
    .long 0                         # DW_AT_location
    .byte 0                         # End Of Children Mark
.LCUEnd:

.section .debug_loc.dwo, "e", @progbits
## Start the section with a number of dummy DW_LLE_end_of_list entries to check
## that the reading offset is correctly adjusted.
    .zero 0x13
.LLocBegin:
    .byte 3                         # DW_LLE_startx_length
    .uleb128 1                      # Index
    .long 0x10                      # Length
    .short 1                        # Loc expr size
    .byte 85                        # DW_OP_reg5
    .byte 0                         # DW_LLE_end_of_list
.LLocEnd:

    .section .debug_cu_index, "", @progbits
## Header:
    .long 2                         # Version
    .long 3                         # Section count
    .long 1                         # Unit count
    .long 2                         # Slot count
## Hash Table of Signatures:
    .quad 0x1100001122222222
    .quad 0
## Parallel Table of Indexes:
    .long 1
    .long 0
## Table of Section Offsets:
## Row 0:
    .long 1                         # DW_SECT_INFO
    .long 3                         # DW_SECT_ABBREV
    .long 5                         # DW_SECT_LOC
## Row 1:
    .long 0                         # Offset in .debug_info.dwo
    .long 0                         # Offset in .debug_abbrev.dwo
    .long .LLocBegin-.debug_loc.dwo # Offset in .debug_loc.dwo
## Table of Section Sizes:
    .long .LCUEnd-.LCUBegin         # Size in .debug_info.dwo
    .long .LAbbrevEnd-.LAbbrevBegin # Size in .debug_abbrev.dwo
    .long .LLocEnd-.LLocBegin       # Size in .debug_loc.dwo
