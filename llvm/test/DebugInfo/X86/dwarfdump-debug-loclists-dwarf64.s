# RUN: llvm-mc %s -filetype obj -triple x86_64 -o - | \
# RUN:   llvm-dwarfdump -v -debug-loclists - | \
# RUN:   FileCheck %s

# CHECK:      .debug_loclists contents:
# CHECK-NEXT: locations list header:
# CHECK-SAME: length = 0x0000000000000028,
# CHECK-SAME: version = 0x0005,
# CHECK-SAME: addr_size = 0x08,
# CHECK-SAME: seg_size = 0x00,
# CHECK-SAME: offset_entry_count = 0x00000002
# CHECK-NEXT: offsets: [
# CHECK-NEXT: 0x0000000000000010 => 0x00000024
# CHECK-NEXT: 0x0000000000000018 => 0x0000002c
# CHECK-NEXT: ]
# CHECK-NEXT: 0x00000024:
# CHECK-NEXT:   DW_LLE_offset_pair (0x0000000000000001, 0x0000000000000002): DW_OP_consts +7, DW_OP_stack_value
# CHECK-NEXT:   DW_LLE_end_of_list ()
# CHECK-EMPTY:
# CHECK-NEXT: 0x0000002c:
# CHECK-NEXT:   DW_LLE_offset_pair (0x0000000000000005, 0x0000000000000007): DW_OP_consts +12, DW_OP_stack_value
# CHECK-NEXT:   DW_LLE_end_of_list ()

    .section .debug_loclists, "", @progbits
    .long 0xffffffff            # DWARF64 mark
    .quad .LLLEnd-.LLLBegin     # Length
.LLLBegin:
    .short 5                    # Version
    .byte 8                     # Address size
    .byte 0                     # Segment selector size
    .long 2                     # Offset entry count
.LLLBase:
    .quad .LLL0-.LLLBase
    .quad .LLL1-.LLLBase
.LLL0:
    .byte 4                     # DW_LLE_offset_pair
    .uleb128 1                  #   starting offset
    .uleb128 2                  #   ending offset
    .byte 3                     # Loc expr size
    .byte 17                    # DW_OP_consts
    .byte 7                     # 7
    .byte 159                   # DW_OP_stack_value
    .byte 0                     # DW_LLE_end_of_list

.LLL1:
    .byte 4                     # DW_LLE_offset_pair
    .uleb128 5                  #   starting offset
    .uleb128 7                  #   ending offset
    .byte 3                     # Loc expr size
    .byte 17                    # DW_OP_consts
    .byte 12                    # 12
    .byte 159                   # DW_OP_stack_value
    .byte 0                     # DW_LLE_end_of_list
.LLLEnd:
