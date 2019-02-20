# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o %t.o
# RUN: llvm-dwarfdump -v %t.o | FileCheck %s

# DW_LLE_startx_length has different `length` encoding in pre-DWARF 5
# and final DWARF 5 versions. This test checks we are able to parse
# the final version which uses ULEB128 and not the U32.

# CHECK:         .debug_loclists contents:
# CHECK-NEXT:    0x00000000: locations list header: length = 0x0000000f, version = 0x0005, addr_size = 0x08, seg_size = 0x00, offset_entry_count = 0x00000000
# CHECK-NEXT:    0x00000000:
# CHECK-NEXT:    Addr idx 1 (w/ length 16): DW_OP_reg5 RDI

.section .debug_loclists,"",@progbits
 .long  .Ldebug_loclist_table_end0-.Ldebug_loclist_table_start0
.Ldebug_loclist_table_start0:
 .short 5         # Version.
 .byte 8          # Address size.
 .byte 0          # Segmen selector size.
 .long 0          # Offset entry count.
 
 .byte 3          # DW_LLE_startx_length
 .byte 0x01       # Index
 .uleb128 0x10    # Length
 .short 1         # Loc expr size
 .byte 85         # DW_OP_reg5
 .byte 0          # DW_LLE_end_of_list
.Ldebug_loclist_table_end0:
