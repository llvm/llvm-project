# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t.dwp
# RUN: not llvm-dwp %t.dwp -o %t 2>&1 | FileCheck %s

## Note: To reach the test point, we need a DWP file with a CU, a CU index
## of version 2 and a TU index of version 5. A valid TU is not required, but
## the .debug_types.dwo section should not be empty.

# CHECK: error: unsupported tu_index version: 5 (only version 2 is supported)

.section .debug_abbrev.dwo, "e", @progbits
.LAbbrevBegin:
    .uleb128 1                      # Abbreviation Code
    .uleb128 17                     # DW_TAG_compile_unit
    .byte 1                         # DW_CHILDREN_no
    .uleb128 0x2131                 # DW_AT_GNU_dwo_id
    .uleb128 7                      # DW_FORM_data8
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
.LCUEnd:

    .section .debug_types.dwo, "e", @progbits
    .space 1

    .section .debug_cu_index, "", @progbits
## Header:
    .long 2                         # Version
    .long 2                         # Section count
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
## Row 1:
    .long 0                         # Offset in .debug_info.dwo
    .long 0                         # Offset in .debug_abbrev.dwo
## Table of Section Sizes:
    .long .LCUEnd-.LCUBegin         # Size in .debug_info.dwo
    .long .LAbbrevEnd-.LAbbrevBegin # Size in .debug_abbrev.dwo

    .section .debug_tu_index, "", @progbits
## Header:
    .short 5                        # Version
    .space 2                        # Padding
    .long 2                         # Section count
    .long 1                         # Unit count
    .long 2                         # Slot count
## Hash Table of Signatures:
    .quad 0x1100003333333333
    .quad 0
## Parallel Table of Indexes:
    .long 1
    .long 0
## Table of Section Offsets:
## Row 0:
    .long 1                         # DW_SECT_INFO
    .long 3                         # DW_SECT_ABBREV
## Row 1:
    .long 0
    .long 0
## Table of Section Sizes:
    .long 1
    .long 1
