## The test checks that ranges for compile units in package files are read
## correctly, i.e. the base offset in the index section is taken into account.

# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o - | \
# RUN:   llvm-dwarfdump -v -debug-info -debug-rnglists - | \
# RUN:   FileCheck %s

# CHECK:      .debug_info.dwo contents:
# CHECK:      Compile Unit:
# CHECK:      DW_TAG_compile_unit [1]
# CHECK-NEXT:   DW_AT_ranges [DW_FORM_rnglistx] (indexed (0x1) rangelist = 0x00000022
# CHECK-NEXT:     [0x00000005, 0x0000000f))

# CHECK:      .debug_rnglists.dwo contents:
# CHECK:      0x00000000: range list header:
# CHECK:      0x0000000d: range list header:
# CHECK-NEXT: offsets: [
# CHECK-NEXT: 0x00000008 => 0x00000021
# CHECK-NEXT: 0x00000009 => 0x00000022
# CHECK-NEXT: ]
# CHECK-NEXT: ranges:
# CHECK-NEXT: 0x00000021: [DW_RLE_end_of_list]
# CHECK-NEXT: 0x00000022: [DW_RLE_offset_pair]:  0x00000005, 0x0000000f => [0x00000005, 0x0000000f)
# CHECK-NEXT: 0x00000025: [DW_RLE_end_of_list]

    .section .debug_abbrev.dwo, "e", @progbits
.LAbbrev:
    .byte 0x01                          # Abbrev code
    .byte 0x11                          # DW_TAG_compile_unit
    .byte 0x00                          # DW_CHILDREN_no
    .byte 0x55                          # DW_AT_ranges
    .byte 0x23                          # DW_FORM_rnglistx
    .byte 0x00                          # EOM(1)
    .byte 0x00                          # EOM(2)
    .byte 0x00                          # EOM(3)
.LAbbrevEnd:
    
    .section .debug_info.dwo, "e", @progbits
.LCU:
    .long .LCUEnd-.LCUVersion           # Length
.LCUVersion:
    .short 5                            # Version
    .byte 5                             # DW_UT_split_compile
    .byte 4                             # Address Size (in bytes)
    .long 0                             # Offset Into Abbrev Section
    .quad 0x1100001122222222            # DWO id
    .uleb128 1                          # Abbrev [1] DW_TAG_compile_unit
    .uleb128 1                          # DW_AT_ranges (DW_FORM_rnglistx)
.LCUEnd:

    .section .debug_rnglists.dwo,"e",@progbits
.LRLT0:
    .long .LRLT0End-.LRLT0Version       # Length
.LRLT0Version:
    .short 5
    .byte 4
    .byte 0
    .long 0
.LRLT0List0:
    .byte 0                             # DW_RLE_end_of_list
.LRLT0End:

.LRLT1:
    .long .LRLT1End-.LRLT1Version
.LRLT1Version:
    .short 5                            # Version
    .byte 4                             # Address size
    .byte 0                             # Segment selector size
    .long 2                             # Offset entry count
.LRLT1Base:
    .long .LRLT1List0-.LRLT1Base
    .long .LRLT1List1-.LRLT1Base
.LRLT1List0:
    .byte 0                             # DW_RLE_end_of_list
.LRLT1List1:
    .byte 4                             # DW_RLE_offset_pair
    .uleb128 5                          # Starting offset
    .uleb128 15                         # Ending offset
    .byte 0                             # DW_RLE_end_of_list
.LRLT1End:

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
    .long 8                             # DW_SECT_RNGLISTS
## Row 1:
    .long 0                             # Offset in .debug_info.dwo
    .long 0                             # Offset in .debug_abbrev.dwo
    .long .LRLT1-.debug_rnglists.dwo    # Offset in .debug_rnglists.dwo
## Table of Section Sizes:
    .long .LCUEnd-.LCU                  # Size in .debug_info.dwo
    .long .LAbbrevEnd-.LAbbrev          # Size in .debug_abbrev.dwo
    .long .LRLT1End-.LRLT1              # Size in .debug_rnglists.dwo
