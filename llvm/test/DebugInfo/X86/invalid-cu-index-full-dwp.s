# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o - | \
# RUN:   llvm-dwarfdump -debug-info - | FileCheck %s

## The CU index has no empty slot and no slot matches the unit's DWO id.
## A hash lookup must give up rather than probe forever, and the unit is then
## found by its offset.

# CHECK:      DW_UT_split_compile
# CHECK-SAME: DWO_id = 0x1100001122222222
# CHECK:      DW_TAG_compile_unit
# CHECK-NEXT:   DW_AT_name ("a.c")

    .section .debug_abbrev.dwo, "e", @progbits
.LAbbrBegin:
    .uleb128 1                  # Abbreviation Code
    .uleb128 17                 # DW_TAG_compile_unit
    .byte 0                     # DW_CHILDREN_no
    .uleb128 3                  # DW_AT_name
    .uleb128 8                  # DW_FORM_string
    .byte 0                     # EOM(1)
    .byte 0                     # EOM(2)
    .byte 0                     # EOM(3)
.LAbbrEnd:

    .section .debug_info.dwo, "e", @progbits
.LCUBegin:
    .long .LCUEnd-.LCUVersion   # Length
.LCUVersion:
    .short 5                    # Version
    .byte 5                     # DW_UT_split_compile
    .byte 8                     # Address size
    .long 0                     # Abbrev offset
    .quad 0x1100001122222222    # DWO id
    .uleb128 1                  # Abbrev [1] DW_TAG_compile_unit
    .asciz "a.c"                # DW_AT_name
.LCUEnd:

    .section .debug_cu_index, "", @progbits
## Header:
    .short 5                    # Version
    .space 2                    # Padding
    .long 2                     # Section count
    .long 1                     # Unit count
    .long 1                     # Slot count
## Hash Table of Signatures:
    .quad 0x3300003344444444
## Parallel Table of Indexes:
    .long 1
## Table of Section Offsets:
## Row 0:
    .long 1                     # DW_SECT_INFO
    .long 3                     # DW_SECT_ABBREV
## Row 1:
    .long .LCUBegin-.debug_info.dwo     # Offset in .debug_info.dwo
    .long .LAbbrBegin-.debug_abbrev.dwo # Offset in .debug_abbrev.dwo
## Table of Section Sizes:
    .long .LCUEnd-.LCUBegin             # Size in .debug_info.dwo
    .long .LAbbrEnd-.LAbbrBegin         # Size in .debug_abbrev.dwo
