## The test checks that llvm-dwp avoids an out of bound access when there is
## an unknown section identifier in an index section. Without the fix, the test
## failed when LLVM is built with UBSan.
## Note that additional sections (.debug_abbrev.dwo, .debug_info.dwo, and
## .debug_types.dwo) are required to reach the test points in the code.

# RUN: llvm-mc -triple x86_64-unknown-linux-gnu %s -filetype=obj -o %t.dwp
# RUN: llvm-dwp %t.dwp -o - | \
# RUN:   llvm-dwarfdump -debug-cu-index -debug-tu-index - | \
# RUN:   FileCheck %s

## Check that all known sections are preserved and no data for unknown section
## identifiers is copied.

# CHECK:      .debug_cu_index contents:
# CHECK-NEXT: version = 2, units = 1, slots = 2
# CHECK:      Index Signature INFO ABBREV
# CHECK-NOT:  Unknown
# CHECK:      -----
# CHECK-NEXT: 1 0x1100002222222222 [0x0000000000000000, 0x0000000000000014) [0x00000000, 0x00000009)
# CHECK-NOT:  [

# CHECK:      .debug_tu_index contents:
# CHECK-NEXT: version = 2, units = 1, slots = 2
# CHECK:      Index Signature TYPES ABBREV
# CHECK-NOT:  Unknown
# CHECK:      -----
# CHECK-NEXT: 2 0x1100003333333333 [0x0000000000000000, 0x0000000000000019) [0x00000009, 0x00000014)
# CHECK-NOT:  [

.section .debug_abbrev.dwo, "e", @progbits
.LCUAbbrevBegin:
    .uleb128 1                              # Abbreviation Code
    .uleb128 0x11                           # DW_TAG_compile_unit
    .byte 0                                 # DW_CHILDREN_no
    .uleb128 0x2131                         # DW_AT_GNU_dwo_id
    .uleb128 7                              # DW_FORM_data8
    .byte 0                                 # EOM(1)
    .byte 0                                 # EOM(2)
    .byte 0                                 # EOM(3)
.LCUAbbrevEnd:

.LTUAbbrevBegin:
    .uleb128 1                              # Abbreviation Code
    .uleb128 0x41                           # DW_TAG_type_unit
    .byte 1                                 # DW_CHILDREN_yes
    .byte 0                                 # EOM(1)
    .byte 0                                 # EOM(2)
    .uleb128 2                              # Abbreviation Code
    .uleb128 0x13                           # DW_TAG_structure_type
    .byte 0                                 # DW_CHILDREN_no
    .byte 0                                 # EOM(1)
    .byte 0                                 # EOM(2)
    .byte 0                                 # EOM(3)
.LTUAbbrevEnd:

    .section .debug_info.dwo, "e", @progbits
.LCUBegin:
    .long .LCUEnd-.LCUVersion               # Length of Unit
.LCUVersion:
    .short 4                                # Version
    .long 0                                 # Abbrev offset
    .byte 8                                 # Address size
    .uleb128 1                              # Abbrev [1] DW_TAG_compile_unit
    .quad 0x1100002222222222                # DW_AT_GNU_dwo_id
.LCUEnd:

    .section .debug_types.dwo, "e", @progbits
.LTUBegin:
    .long .LTUEnd-.LTUVersion               # Length of Unit
.LTUVersion:
    .short 4                                # Version
    .long 0                                 # Abbrev offset
    .byte 8                                 # Address size
    .quad 0x1100003333333333                # Type signature
    .long .LTUType-.LTUBegin                # Type offset
    .uleb128 1                              # Abbrev [1] DW_TAG_type_unit
.LTUType:
    .uleb128 2                              # Abbrev [2] DW_TAG_structure_type
.LTUEnd:

    .section .debug_cu_index, "", @progbits
## Header:
    .long 2                                 # Version
    .long 4                                 # Section count
    .long 1                                 # Unit count
    .long 2                                 # Slot count
## Hash Table of Signatures:
    .quad 0x1100002222222222
    .quad 0
## Parallel Table of Indexes:
    .long 1
    .long 0
## Table of Section Offsets:
## Row 0:
    .long 1                                 # DW_SECT_INFO
    .long 3                                 # DW_SECT_ABBREV
    .long 0                                 # Invalid ID, less than DW_SECT_INFO
    .long 9                                 # Invalid ID, greater than DW_SECT_MACRO
## Row 1:
    .long 0                                 # Offset in .debug_info.dwo
    .long 0                                 # Offset in .debug_abbrev.dwo
    .long 0
    .long 0
## Table of Section Sizes:
    .long .LCUEnd-.LCUBegin                 # Size in .debug_info.dwo
    .long .LCUAbbrevEnd-.LCUAbbrevBegin     # Size in .debug_abbrev.dwo
    .long 1
    .long 1

    .section .debug_tu_index, "", @progbits
## Header:
    .long 2                                 # Version
    .long 4                                 # Section count
    .long 1                                 # Unit count
    .long 2                                 # Slot count
## Hash Table of Signatures:
    .quad 0
    .quad 0x1100003333333333
## Parallel Table of Indexes:
    .long 0
    .long 1
## Table of Section Offsets:
## Row 0:
    .long 2                                 # DW_SECT_TYPES
    .long 3                                 # DW_SECT_ABBREV
    .long 0                                 # Invalid ID, less than DW_SECT_INFO
    .long 9                                 # Invalid ID, greater than DW_SECT_MACRO
## Row 1:
    .long 0                                 # Offset in .debug_types.dwo
    .long .LTUAbbrevBegin-.debug_abbrev.dwo # Offset in .debug_abbrev.dwo
    .long 0
    .long 0
## Table of Section Sizes:
    .long .LTUEnd-.LTUBegin                 # Size in .debug_types.dwo
    .long .LTUAbbrevEnd-.LTUAbbrevBegin     # Size in .debug_abbrev.dwo
    .long 1
    .long 1
