# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o - | \
# RUN:   llvm-dwarfdump -debug-cu-index - | FileCheck %s

## Two units are declared, but the parallel table only refers to unit 1.
## Unit 2 has no contribution and must be rejected rather than dereferenced.

# CHECK: .debug_cu_index contents:
# CHECK-NOT: 0x

    .section .debug_cu_index, "", @progbits
## Header:
    .long 2             # Version
    .long 1             # Section count
    .long 2             # Unit count
    .long 1             # Slot count
## Hash Table of Signatures:
    .quad 0x1111111111111111
## Parallel Table of Indexes:
    .long 1
## Table of Section Offsets:
    .long 1             # DW_SECT_INFO
    .long 0x100         # Unit 1 offset
    .long 0x200         # Unit 2 offset
## Table of Section Sizes:
    .long 0x10          # Unit 1 size
    .long 0x20          # Unit 2 size
