# RUN: llvm-mc %s -filetype obj -triple x86_64-pc-linux -o - | \
# RUN:   llvm-dwarfdump -debug-addr - | \
# RUN:   FileCheck %s

## This checks that we use DWARFDataExtractor::getRelocatedAddress() to read
## addresses of an address table. In this test, the raw data in the .debug_addr
## section does not contain the full address, thus, it is required to resolve
## a RELA relocation to recover the real value.

# CHECK:      .debug_addr contents
# CHECK-NEXT: length = 0x0000000c, format = DWARF32, version = 0x0005, addr_size = 0x08, seg_size = 0x00
# CHECK-NEXT: Addrs: [
# CHECK-NEXT: 0x000000000000002a
# CHECK-NEXT: ]

    .text
    .space  0x2a
.Lfoo:

    .section .debug_addr,"",@progbits
    .long   .LAddr0end-.LAddr0version   # Length
.LAddr0version:
    .short  5                           # Version
    .byte   8                           # Address size
    .byte   0                           # Segment selector size
.LAddr0table:
    .quad   .Lfoo
.LAddr0end:
