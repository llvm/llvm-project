## The test checks that we can read DWARFv5 type units in DWP files.

# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o - | \
# RUN:   llvm-dwarfdump -debug-info - | \
# RUN:   FileCheck %s

# CHECK:      .debug_info.dwo contents:
# CHECK:      Type Unit:
# CHECK:      DW_TAG_type_unit
# CHECK-NEXT:   DW_AT_visibility (DW_VIS_local)
# CHECK:      DW_TAG_structure_type
# CHECK-NEXT:   DW_AT_name ("foo")

    .section .debug_abbrev.dwo, "e", @progbits
## Reserve some space in the section so that the abbreviation table for the type
## unit does not start at the beginning of the section and thus the base offset
## from the index section should be added to find the correct offset.
    .space 16
.LAbbrevBegin:
    .uleb128 1                              # Abbrev code
    .uleb128 0x41                           # DW_TAG_type_unit
    .byte 1                                 # DW_CHILDREN_yes
    .uleb128 0x17                           # DW_AT_visibility
    .uleb128 0x0b                           # DW_FORM_data1
    .byte 0                                 # EOM(1)
    .byte 0                                 # EOM(2)
    .uleb128 2                              # Abbrev code
    .uleb128 0x13                           # DW_TAG_structure_type
    .byte 0                                 # DW_CHILDREN_no
    .uleb128 0x03                           # DW_AT_name
    .uleb128 0x08                           # DW_FORM_string
    .byte 0                                 # EOM(1)
    .byte 0                                 # EOM(2)
    .byte 0                                 # EOM(3)
.LAbbrevEnd:

    .section .debug_info.dwo, "e", @progbits
.LTUBegin:
    .long .LTUEnd-.LTUVersion               # Length of Unit
.LTUVersion:
    .short 5                                # DWARF version number
    .byte 6                                 # DW_UT_split_type
    .byte 8                                 # Address Size (in bytes)
    .long 0                                 # Offset Into Abbrev. Section
    .quad 0x1100001122222222                # Type Signature
    .long .LTUType-.LTUBegin                # Type offset
    .uleb128 1                              # Abbrev [1] DW_TAG_type_unit
    .byte 1                                 # DW_AT_visibility
.LTUType:
    .uleb128 2                              # Abbrev [2] DW_TAG_structure_type
    .asciz "foo"                            # DW_AT_name
.LTUEnd:

    .section .debug_tu_index, "", @progbits
## Header:
    .short 5                                # Version
    .space 2                                # Padding
    .long 2                                 # Section count
    .long 1                                 # Unit count
    .long 2                                 # Slot count
## Hash Table of Signatures:
    .quad 0x1100001122222222
    .quad 0
## Parallel Table of Indexes:
    .long 1
    .long 0
## Table of Section Offsets:
## Row 0:
    .long 1                                 # DW_SECT_INFO
    .long 3                                 # DW_SECT_ABBREV
## Row 1:
    .long 0                                 # Offset in .debug_info.dwo
    .long .LAbbrevBegin-.debug_abbrev.dwo   # Offset in .debug_abbrev.dwo
## Table of Section Sizes:
    .long .LTUEnd-.LTUBegin                 # Size in .debug_info.dwo
    .long .LAbbrevEnd-.LAbbrevBegin         # Size in .debug_abbrev.dwo
