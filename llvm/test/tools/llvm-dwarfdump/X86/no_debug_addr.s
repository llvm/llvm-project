# RUN: llvm-mc %s -filetype obj -triple=x86_64-pc-linux -o %t.o
# RUN: llvm-dwarfdump -v %t.o | FileCheck %s -match-full-lines

## Ensure bogus empty section names are not printed when dumping
## rnglists that reference debug_addr when it is not present (such as in .dwo files)

# CHECK:       DW_AT_ranges [DW_FORM_rnglistx]   (indexed (0x0) rangelist = 0x00000004
# CHECK-NEXT:    [0x0000000000000000, 0x0000000000000001)
# CHECK-NEXT:    [0x0000000000000000, 0x0000000000000002))

.section .debug_info.dwo,"e",@progbits
.long .Ldebug_info_dwo_end1-.Ldebug_info_dwo_start1   # Length of Unit
.Ldebug_info_dwo_start1:
  .short 5                       # DWARF version number
  .byte  5                       # DWARF Unit Type
  .byte  8                       # Address Size (in bytes)
  .long  0                       # Offset Into Abbrev. Section
  .quad  -6809755978868859807
  .byte  1                       # Abbrev [1] 0x14:0x32 DW_TAG_compile_unit
  .byte  0                       # DW_AT_ranges
.Ldebug_info_dwo_end1:

.section .debug_abbrev.dwo,"e",@progbits
  .byte 1                        # Abbreviation Code
  .byte 17                       # DW_TAG_compile_unit
  .byte 0                        # DW_CHILDREN_no
  .byte 85                       # DW_AT_ranges
  .byte 35                       # DW_FORM_rnglistx
  .byte 0                        # EOM(1)
  .byte 0                        # EOM(2)

.section .debug_rnglists.dwo,"e",@progbits
  .long  .Ldebug_rnglist_table_end1-.Ldebug_rnglist_table_start1 # Length
.Ldebug_rnglist_table_start1:
  .short  5                      # Version
  .byte  8                       # Address size
  .byte  0                       # Segment selector size
  .long  1                       # Offset entry count
.Lrnglists_dwo_table_base0:
  .long  .Ldebug_ranges0-.Lrnglists_dwo_table_base0
.Ldebug_ranges0:
  .byte  1                       # DW_RLE_base_addressx
  .byte  0                       #   base address index
  .byte  4                       # DW_RLE_offset_pair
  .byte  0                       #   starting offset
  .byte  1                       #   ending offset
  .byte  3                       # DW_RLE_startx_length
  .byte  1                       #   start index
  .byte  2                       #   length
  .byte  0                       # DW_RLE_end_of_list
.Ldebug_rnglist_table_end1:
