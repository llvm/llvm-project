# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o - | \
# RUN:   llvm-dwarfdump -debug-addr - | \
# RUN:   FileCheck %s

# CHECK:      .debug_addr contents:
# CHECK-NEXT: Address table header:
# CHECK-SAME: length = 0x000000000000000c,
# CHECK-SAME: format = DWARF64,
# CHECK-SAME: version = 0x0005,
# CHECK-SAME: addr_size = 0x04,
# CHECK-SAME: seg_size = 0x00
# CHECK-NEXT: Addrs: [
# CHECK-NEXT: 0x00000000
# CHECK-NEXT: 0x00001000
# CHECK-NEXT: ]

    .section .debug_addr,"",@progbits
    .long 0xffffffff                    # DWARF64 mark
    .quad .LAddr0end-.LAddr0version     # Length
.LAddr0version:
    .short 5                            # Version
    .byte 4                             # Address size
    .byte 0                             # Segment selector size
    .long 0x00000000
    .long 0x00001000
.LAddr0end:
