# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o - | \
# RUN:   llvm-dwarfdump -debug-cu-index - | \
# RUN:   FileCheck %s

# CHECK:      .debug_cu_index contents:
# CHECK-NEXT: version = 2 slots = 2
# CHECK-EMPTY:
# CHECK-NEXT: Index Signature          INFO                     <unknown>
# CHECK-NEXT: ----- ------------------ ------------------------ ------------------------
# CHECK-NEXT:     1 0x1100001122222222 [0x00001000, 0x00001010) [0x00002000, 0x00002020)

    .section .debug_cu_index, "", @progbits
## Header:
    .long 2             # Version
    .long 2             # Section count
    .long 1             # Unit count
    .long 2             # Slot count
## Hash Table of Signatures:
    .quad 0x1100001122222222
    .quad 0
## Parallel Table of Indexes:
    .long 1
    .long 0
## Table of Section Offsets:
## Row 0:
    .long 1             # DW_SECT_INFO
    .long 9             # Invalid section identifier
## Row 1:
    .long 0x1000        # Offset in .debug_info.dwo
    .long 0x2000
## Table of Section Sizes:
    .long 0x10          # Size in .debug_info.dwo
    .long 0x20
