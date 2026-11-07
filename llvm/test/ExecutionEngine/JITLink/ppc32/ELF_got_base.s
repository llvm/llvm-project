# RUN: llvm-mc -triple=powerpc-linux-gnu -filetype=obj %s -o %t.o
# RUN: llvm-jitlink -noexec -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 -check %s %t.o
# RUN: llvm-mc -triple=powerpcle-linux-gnu -filetype=obj %s -o %t.o
# RUN: llvm-jitlink -noexec -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 -check %s %t.o

# A GOT base reference must be resolved even when there are no GOT entries.
        .text
        .globl main
main:
        lis 3, _GLOBAL_OFFSET_TABLE_@ha
        addi 3, 3, _GLOBAL_OFFSET_TABLE_@l
        blr
# jitlink-check: (*{4}main) & 0xffff = ((_GLOBAL_OFFSET_TABLE_ + 0x8000) >> 16) & 0xffff
# jitlink-check: (*{4}(main+4)) & 0xffff = _GLOBAL_OFFSET_TABLE_ & 0xffff

        .data
        .globl got_base
got_base:
        .long _GLOBAL_OFFSET_TABLE_
# jitlink-check: *{4}got_base = _GLOBAL_OFFSET_TABLE_
