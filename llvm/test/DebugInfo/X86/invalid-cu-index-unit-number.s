# RUN: llvm-mc -triple x86_64-unknown-linux %s -filetype=obj -o - | \
# RUN:   llvm-dwarfdump -debug-cu-index - | FileCheck %s

## The parallel table refers to unit 2, but the index only declares one unit.
## The parser must reject the index instead of writing past the Contribs array.

# CHECK: .debug_cu_index contents:
# CHECK-NOT: 0x

    .section .debug_cu_index, "", @progbits
## Header:
    .long 2             # Version
    .long 1             # Section count
    .long 1             # Unit count
    .long 2             # Slot count
## Hash Table of Signatures:
    .quad 0
    .quad 0xdeadbeefcafebabe
## Parallel Table of Indexes:
    .long 0
    .long 2
## Table of Section Offsets:
    .long 1             # DW_SECT_INFO
    .long 0x100
## Table of Section Sizes:
    .long 0x10
