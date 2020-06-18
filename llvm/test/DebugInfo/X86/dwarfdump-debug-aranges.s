# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o - | \
# RUN:  llvm-dwarfdump -debug-aranges - 2>&1 | \
# RUN:  FileCheck %s

    .section .debug_aranges,"",@progbits
# CHECK: .debug_aranges contents:

## Case 1: Check that an empty set of ranges is supported.
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

## Case 2: Check that the address size of 4 is supported.
    .long   .L2end - .L2version     # Length
# CHECK: Address Range Header: length = 0x0000001c,
.L2version:
    .short  2                       # Version
    .long   0x112233                # Debug Info Offset
    .byte   4                       # Address Size
    .byte   0                       # Segment Selector Size
# CHECK-SAME: version = 0x0002,
# CHECK-SAME: cu_offset = 0x00112233,
# CHECK-SAME: addr_size = 0x04,
# CHECK-SAME: seg_size = 0x00
    .space  4                       # Padding
.L2tuples:
    .long   0x11223344, 0x01020304  # Address and length
# CHECK-NEXT: [0x11223344,  0x12243648)
    .long   0, 0                    # Termination tuple
# CHECK-NOT: [0x
.L2end:

## Case 3: Check that the address size of 8 is also supported.
    .long   .L3end - .L3version     # Length
# CHECK: Address Range Header: length = 0x0000002c,
.L3version:
    .short  2                       # Version
    .long   0x112233                # Debug Info Offset
    .byte   8                       # Address Size
    .byte   0                       # Segment Selector Size
# CHECK-SAME: version = 0x0002,
# CHECK-SAME: cu_offset = 0x00112233,
# CHECK-SAME: addr_size = 0x08,
# CHECK-SAME: seg_size = 0x00
    .space  4                       # Padding
.L3tuples:
    .quad   0x1122334455667788      # Address
    .quad   0x0102030405060708      # Length
# CHECK-NEXT: [0x1122334455667788,  0x122436485a6c7e90)
    .quad   0, 0                    # Termination tuple
# CHECK-NOT: [0x
.L3end:

## Case 4: Check that 64-bit DWARF format is supported.
    .long 0xffffffff                # DWARF64 mark
    .quad   .L4end - .L4version     # Length
# CHECK: Address Range Header: length = 0x000000000000001c,
# CHECK-SAME: format = DWARF64,
.L4version:
    .short  2                       # Version
    .quad   0x123456789abc          # Debug Info Offset
    .byte   4                       # Address Size
    .byte   0                       # Segment Selector Size
# CHECK-SAME: version = 0x0002,
# CHECK-SAME: cu_offset = 0x0000123456789abc,
# CHECK-SAME: addr_size = 0x04,
# CHECK-SAME: seg_size = 0x00
                                    # No padding
.L4tuples:
    .long   0, 1                    # Address and length
# CHECK-NEXT: [0x00000000,  0x00000001)
    .long   0, 0                    # Termination tuple
# CHECK-NOT: [0x
.L4end:
