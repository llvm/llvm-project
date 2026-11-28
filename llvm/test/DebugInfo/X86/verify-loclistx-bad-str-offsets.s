# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o %t
# RUN: not llvm-dwarfdump --verify %t 2>&1 | FileCheck %s

## DW_AT_str_offsets_base points past the end of a 4 byte .debug_str_offsets
## section, so tryExtractDIEsIfNeeded() reports an error and returns before it
## reaches the end of the function. Everything derived from the unit DIE must
## already be in place at that point: the loclistx and rnglistx lookups below
## are still verified afterwards and used to dereference an unset location or
## range table.
##
## The CHECK-NOT lines matter: they prove both lookups resolved successfully
## after the string offsets failure, rather than merely that the verifier did
## not crash before it reached .debug_line.

# CHECK: error: invalid reference to or invalid content in .debug_str_offsets
# CHECK-NOT: error: Loclist table not found
# CHECK-NOT: error: invalid range list table
# CHECK: Verifying .debug_line...
# CHECK: Errors detected.

    .section .debug_abbrev, "", @progbits
.LAbbrevBegin:
    .uleb128 1                  # Abbreviation code
    .uleb128 0x11               # DW_TAG_compile_unit
    .byte 1                     # DW_CHILDREN_yes
    .uleb128 0x72               # DW_AT_str_offsets_base
    .uleb128 0x17               # DW_FORM_sec_offset
    .uleb128 0x8c               # DW_AT_loclists_base
    .uleb128 0x17               # DW_FORM_sec_offset
    .uleb128 0x8b               # DW_AT_rnglists_base
    .uleb128 0x17               # DW_FORM_sec_offset
    .byte 0                     # EOM(1)
    .byte 0                     # EOM(2)
    .uleb128 2                  # Abbreviation code
    .uleb128 0x34               # DW_TAG_variable
    .byte 0                     # DW_CHILDREN_no
    .uleb128 2                  # DW_AT_location
    .uleb128 0x22               # DW_FORM_loclistx
    .byte 0                     # EOM(1)
    .byte 0                     # EOM(2)
    .uleb128 3                  # Abbreviation code
    .uleb128 0x2e               # DW_TAG_subprogram
    .byte 0                     # DW_CHILDREN_no
    .uleb128 0x55               # DW_AT_ranges
    .uleb128 0x23               # DW_FORM_rnglistx
    .byte 0                     # EOM(1)
    .byte 0                     # EOM(2)
    .byte 0                     # EOM(3)
.LAbbrevEnd:

    .section .debug_info, "", @progbits
.LCUBegin:
    .long .LCUEnd-.LCUVersion   # Length of unit
.LCUVersion:
    .short 5                    # Version
    .byte 1                     # DW_UT_compile
    .byte 8                     # Address size
    .long 0                     # Abbrev offset
    .uleb128 1                  # Abbrev [1] DW_TAG_compile_unit
    .long 0x1000                # DW_AT_str_offsets_base, past the section
    .long .LLLBase-.debug_loclists # DW_AT_loclists_base
    .long .LRBase-.debug_rnglists  # DW_AT_rnglists_base
    .uleb128 2                  # Abbrev [2] DW_TAG_variable
    .uleb128 0                  # DW_AT_location (DW_FORM_loclistx) 0
    .uleb128 3                  # Abbrev [3] DW_TAG_subprogram
    .uleb128 0                  # DW_AT_ranges (DW_FORM_rnglistx) 0
    .byte 0                     # End of children mark
.LCUEnd:

    .section .debug_str_offsets, "", @progbits
    .long 0

    .section .debug_loclists, "", @progbits
.LLLBegin:
    .long .LLLEnd-.LLLVersion   # Length of unit
.LLLVersion:
    .short 5                    # Version
    .byte 8                     # Address size
    .byte 0                     # Segment selector size
    .long 1                     # Offset entry count
.LLLBase:
    .long .LLL0-.LLLBase
.LLL0:
    .byte 8                     # DW_LLE_start_length
    .quad 0                     # Start address
    .uleb128 0x10               # Length
    .uleb128 1                  # Location expression size
    .byte 85                    # DW_OP_reg5
    .byte 0                     # DW_LLE_end_of_list
.LLLEnd:

    .section .debug_rnglists, "", @progbits
.LRLBegin:
    .long .LRLEnd-.LRLVersion   # Length of unit
.LRLVersion:
    .short 5                    # Version
    .byte 8                     # Address size
    .byte 0                     # Segment selector size
    .long 1                     # Offset entry count
.LRBase:
    .long .LRL0-.LRBase
.LRL0:
    .byte 7                     # DW_RLE_start_length
    .quad 0                     # Start address
    .uleb128 0x10               # Length
    .byte 0                     # DW_RLE_end_of_list
.LRLEnd:
