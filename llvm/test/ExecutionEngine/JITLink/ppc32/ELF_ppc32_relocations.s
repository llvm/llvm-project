# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=powerpc-unknown-linux-gnu -filetype=obj %s -o %t/main.o
# RUN: llvm-mc -triple=powerpc-unknown-linux-gnu -filetype=obj %S/Inputs/ELF_ppc32_target.s -o %t/target.o
# RUN: llvm-jitlink -noexec -entry=main -abs target_addend=0x12345678 -abs abs_target=0x1000 -abs zero_sym=0 -abs high_sym=0xf0000000 -slab-address=0x10008000 -slab-allocate=128Kb -slab-page-size=4096 -check %s %t/main.o %t/target.o
# RUN: mkdir -p %t/le
# RUN: llvm-mc -triple=powerpcle-unknown-linux-gnu -filetype=obj %s -o %t/le/main.o
# RUN: llvm-mc -triple=powerpcle-unknown-linux-gnu -filetype=obj %S/Inputs/ELF_ppc32_target.s -o %t/le/target.o
# RUN: llvm-jitlink -noexec -entry=main -abs target_addend=0x12345678 -abs abs_target=0x1000 -abs zero_sym=0 -abs high_sym=0xf0000000 -slab-address=0x10008000 -slab-allocate=128Kb -slab-page-size=4096 -check %s %t/le/main.o %t/le/target.o

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

# R_PPC_GOT16 and its split forms.
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

# GOT entries include relocation addends.
        .text
        .globl got_addend
got_addend:
        lwz 3, value2+4@got(30)
# jitlink-check: (*{4}got_addend) & 0xffff = (got_addr(main.o, value2) - _GLOBAL_OFFSET_TABLE_) & 0xffff
# jitlink-check: *{4}got_addr(main.o, value2) = value2 + 4

        .data
        .globl value2
value2:
        .long 0, 0

# R_PPC_LOCAL24PC branches to blrl at _GLOBAL_OFFSET_TABLE_ - 4.
        .text
        .globl local24pc
local24pc:
        bl _GLOBAL_OFFSET_TABLE_@local-4
# jitlink-check: (*{4}local24pc) & 0x03fffffc = ((_GLOBAL_OFFSET_TABLE_ - 4) - local24pc) & 0x03fffffc
# jitlink-check: *{4}(_GLOBAL_OFFSET_TABLE_ - 4) = 0x4e800021

# GOT pointer setup with R_PPC_REL16_HA/LO.
        .globl rel16_got
rel16_got:
        bcl 20, 31, 1f
1:      mflr 30
        addis 30, 30, _GLOBAL_OFFSET_TABLE_-1b@ha
        addi 30, 30, _GLOBAL_OFFSET_TABLE_-1b@l
# jitlink-check: (*{4}(rel16_got+8)) & 0xffff = (((_GLOBAL_OFFSET_TABLE_ - (rel16_got+4)) + 0x8000) >> 16) & 0xffff
# jitlink-check: (*{4}(rel16_got+12)) & 0xffff = (_GLOBAL_OFFSET_TABLE_ - (rel16_got+4)) & 0xffff
        .globl rel16
rel16:
        .reloc ., R_PPC_REL16, rel16_got
        .short 0
        .short 0
# jitlink-check: *{2}rel16 = (rel16_got - rel16) & 0xffff

# The PLTREL24 addend selects the .got2 base.
        .globl pltrel24
pltrel24:
        bl target+32768@plt
# jitlink-check: (*{4}pltrel24) & 0x03fffffc = (stub_addr(main.o, target) - pltrel24) & 0x03fffffc

# Conditional and absolute branches.
        .globl branches
branches:
        beq 0, rel14_target
        ba abs_target
        bca 12, 2, abs_target
        .globl rel14_target
rel14_target:
        blr
# jitlink-check: (*{4}branches) & 0xfffc = (rel14_target - branches) & 0xfffc
# jitlink-check: (*{4}(branches+4)) & 0x03fffffc = abs_target
# jitlink-check: (*{4}(branches+8)) & 0xfffc = abs_target

# EH-frame relocations.
        .globl with_cfi
        .type with_cfi,@function
with_cfi:
        .cfi_startproc
        blr
        .cfi_endproc
        .size with_cfi, .-with_cfi

# Unaligned and 16-bit data relocations.
        .data
        .globl unaligned
unaligned:
        .byte 0
        .reloc ., R_PPC_UADDR32, target
        .long 0
        .reloc ., R_PPC_UADDR16, abs_target
        .short 0
        .short abs_target
# jitlink-check: *{4}(unaligned+1) = target
# jitlink-check: *{2}(unaligned+5) = abs_target
# jitlink-check: *{2}(unaligned+7) = abs_target

# R_PPC_ADDR32 with a negative addend.
        .p2align 2
        .globl negative
negative:
        .long zero_sym - 4
# jitlink-check: *{4}negative = 0xfffffffc

# R_PPC_REL32 wraps modulo 2^32.
        .globl wrap_delta
wrap_delta:
        .long high_sym - wrap_delta
# jitlink-check: *{4}wrap_delta = (high_sym - wrap_delta) & 0xffffffff
