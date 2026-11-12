# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o - | \
# RUN:   llvm-dwarfdump -debug-info - | \
# RUN:   FileCheck %s

# CHECK:      .debug_info.dwo contents:

# CHECK:      0x00000000: Compile Unit:
# CHECK-SAME:   length = 0x0000000000000018
# CHECK-SAME:   version = 0x0004
# CHECK-SAME:   abbr_offset = 0x0000
# CHECK-SAME:   addr_size = 0x04

# CHECK:      0x00000017: DW_TAG_compile_unit
# CHECK-NEXT:   DW_AT_name ("a.c")
# CHECK-NEXT:   DW_AT_GNU_dwo_id (0x1100001122222222)

    .section .debug_abbrev.dwo, "e", @progbits
.LAbbrBegin:
    .uleb128 1                  # Abbreviation Code
    .uleb128 17                 # DW_TAG_compile_unit
    .byte 0                     # DW_CHILDREN_no
    .uleb128 3                  # DW_AT_name
    .uleb128 8                  # DW_FORM_string
    .uleb128 0x2131             # DW_AT_GNU_dwo_id
    .uleb128 7                  # DW_FORM_data8
    .byte 0                     # EOM(1)
    .byte 0                     # EOM(2)
    .byte 0                     # EOM(3)
.LAbbrEnd:

    .section .debug_info.dwo, "e", @progbits
.LCUBegin:
    .long 0xffffffff            # DWARF64 mark
    .quad .LCUEnd-.LCUVersion   # Length
.LCUVersion:
    .short 4                    # Version
    .quad 0                     # Abbrev offset
    .byte 4                     # Address size
    .uleb128 1                  # Abbrev [1] DW_TAG_compile_unit
    .asciz "a.c"                # DW_AT_name
    .quad 0x1100001122222222    # DW_AT_GNU_dwo_id
.LCUEnd:

    .section .debug_cu_index, "", @progbits
## Header:
    .short 2                    # Version
    .space 2                    # Padding
    .long 2                     # Section count
    .long 1                     # Unit count
    .long 4                     # Slot count
## Hash Table of Signatures:
    .quad 0
    .quad 0
    .quad 0x1100001122222222
    .quad 0
## Parallel Table of Indexes:
    .long 0
    .long 0
    .long 1
    .long 0
## Table of Section Offsets:
## Row 0:
    .long 1                     # DW_SECT_INFO
    .long 3                     # DW_SECT_ABBREV
## Row 1:
    .long .LCUBegin-.debug_info.dwo     # Offset in .debug_info.dwo
    .long .LAbbrBegin-.debug_abbrev.dwo # Offset in .debug_abbrev.dwo
## Table of Section Sizes:
    .long .LCUEnd-.LCUBegin
    .long .LAbbrEnd-.LAbbrBegin
