# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o - | \
# RUN:  llvm-dwarfdump -debug-aranges - 2>&1 | \
# RUN:  FileCheck %s

    .section .debug_aranges,"",@progbits
# CHECK: .debug_aranges contents:

## Check that an empty set of ranges is supported.
    .long   .L1end - .L1version     # Length
# CHECK: Address Range Header: length = 0x00000014,
.L1version:
    .short  2                       # Version
    .long   0x3456789a              # Debug Info Offset
    .byte   4                       # Address Size
    .byte   0                       # Segment Selector Size
# CHECK-SAME: version = 0x0002,
# CHECK-SAME: cu_offset = 0x3456789a,
# CHECK-SAME: addr_size = 0x04,
# CHECK-SAME: seg_size = 0x00
    .space 4                        # Padding
.L1tuples:
    .long   0, 0                    # Termination tuple
# CHECK-NOT: [0x
.L1end:
