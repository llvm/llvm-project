# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=powerpc-unknown-linux-gnu -filetype=obj %s -o %t/main.o
# RUN: llvm-mc -triple=powerpc-unknown-linux-gnu -filetype=obj %S/Inputs/ELF_ppc32_target.s -o %t/target.o
# RUN: llvm-jitlink -noexec -entry=main -abs target_addend=0x12345678 -slab-address=0x10008000 -slab-allocate=128Kb -slab-page-size=4096 -check %s %t/main.o %t/target.o
# RUN: mkdir -p %t/le
# RUN: llvm-mc -triple=powerpcle-unknown-linux-gnu -filetype=obj %s -o %t/le/main.o
# RUN: llvm-mc -triple=powerpcle-unknown-linux-gnu -filetype=obj %S/Inputs/ELF_ppc32_target.s -o %t/le/target.o
# RUN: llvm-jitlink -noexec -entry=main -abs target_addend=0x12345678 -slab-address=0x10008000 -slab-allocate=128Kb -slab-page-size=4096 -check %s %t/le/main.o %t/le/target.o

        .text
        .globl main
        .type main,@function
main:
        lis 3, target@ha
        addi 3, 3, target@l
# jitlink-check: decode_operand(main, 1) = (target + 0x8000) >> 16
# jitlink-check: decode_operand(main+4, 2) & 0xffff = target & 0xffff
        .globl branch
branch:
        bl target
# jitlink-check: (*{4}branch) & 0x03fffffc = (stub_addr(main.o, target) - branch) & 0x03fffffc
# jitlink-check: (*{4}got_addr(main.o, target)) = target
# jitlink-check: (*{4}stub_addr(main.o, target)) & 0xffff = ((got_addr(main.o, target) + 0x8000) >> 16) & 0xffff
# jitlink-check: (*{4}(stub_addr(main.o, target) + 4)) & 0xffff = got_addr(main.o, target) & 0xffff
        blr
        .size main, .-main

        .data
        .globl pointer
pointer:
        .long target
# jitlink-check: (*{4}pointer) = target
        .globl delta
delta:
        .long target - delta
# jitlink-check: (*{4}delta) = target - delta

# Check R_PPC_GOT16 and its split forms against the biased GOT base.
        .text
        .globl got_loads
got_loads:
        lwz 3, value@got(30)
        addis 4, 30, value@got@ha
        lwz 4, value@got@l(4)
        lis 5, value@got@h
        blr
# jitlink-check: (*{4}got_loads) & 0xffff = (got_addr(main.o, value) - _GLOBAL_OFFSET_TABLE_) & 0xffff
# jitlink-check: (*{4}(got_loads+4)) & 0xffff = ((got_addr(main.o, value) - _GLOBAL_OFFSET_TABLE_ + 0x8000) >> 16) & 0xffff
# jitlink-check: (*{4}(got_loads+8)) & 0xffff = (got_addr(main.o, value) - _GLOBAL_OFFSET_TABLE_) & 0xffff
# jitlink-check: (*{4}(got_loads+12)) & 0xffff = ((got_addr(main.o, value) - _GLOBAL_OFFSET_TABLE_) >> 16) & 0xffff
# jitlink-check: *{4}got_addr(main.o, value) = value

        .data
        .globl value
value:
        .long 42
        .globl got_base
got_base:
        .long _GLOBAL_OFFSET_TABLE_
# jitlink-check: *{4}got_base = _GLOBAL_OFFSET_TABLE_

# A non-zero external branch addend belongs in the GOT pointer, not the stub PC.
        .text
        .globl branch_addend
branch_addend:
        bl target_addend+4
# jitlink-check: (*{4}branch_addend) & 0x03fffffc = (stub_addr(main.o, target_addend) - branch_addend) & 0x03fffffc
# jitlink-check: *{4}got_addr(main.o, target_addend) = target_addend + 4
