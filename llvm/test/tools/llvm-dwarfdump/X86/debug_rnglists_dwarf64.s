# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o - | \
# RUN:   llvm-dwarfdump -v --debug-rnglists - | \
# RUN:   FileCheck %s

# CHECK:      .debug_rnglists contents:
# CHECK-NEXT: 0x00000000: range list header:
# CHECK-SAME:   length = 0x000000000000001a,
# CHECK-SAME:   format = DWARF64,
# CHECK-SAME:   version = 0x0005,
# CHECK-SAME:   addr_size = 0x08,
# CHECK-SAME:   seg_size = 0x00,
# CHECK-SAME:   offset_entry_count = 0x00000002
# CHECK-NEXT: offsets: [
# CHECK-NEXT: 0x0000000000000010 => 0x00000024
# CHECK-NEXT: 0x0000000000000011 => 0x00000025
# CHECK-NEXT: ]
# CHECK-NEXT: ranges:
# CHECK-NEXT: 0x00000024: [DW_RLE_end_of_list]
# CHECK-NEXT: 0x00000025: [DW_RLE_end_of_list]

    .section .debug_rnglists,"",@progbits
    .long 0xffffffff         # DWARF64 mark
    .quad .Lend - .Lversion  # Table length
.Lversion:
    .short 5                 # Version
    .byte 8                  # Address size
    .byte 0                  # Segment selector size
    .long 2                  # Offset entry count

.Loffsets:
    .quad .Ltable0 - .Loffsets
    .quad .Ltable1 - .Loffsets

.Ltable0:
    .byte 0          # DW_RLE_end_of_list

.Ltable1:
    .byte 0          # DW_RLE_end_of_list

.Lend:
