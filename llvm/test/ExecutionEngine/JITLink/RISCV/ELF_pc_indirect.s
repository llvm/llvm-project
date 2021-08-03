# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=riscv64 -position-independent -filetype=obj -o %t/elf_riscv64_sm_pic_reloc.o %s
# RUN: llvm-mc -triple=riscv32 -position-independent -filetype=obj -o %t/elf_riscv32_sm_pic_reloc.o %s
# RUN: llvm-jitlink -noexec -slab-allocate 100Kb -slab-address 0xfff00000 \
# RUN:              -check %s %t/elf_riscv64_sm_pic_reloc.o
# RUN: llvm-jitlink -noexec -slab-allocate 100Kb -slab-address 0xfff00000 \
# RUN:              -check %s %t/elf_riscv32_sm_pic_reloc.o
#
# Test ELF small/PIC relocations

        .text
        .file   "testcase.c"

# Empty main entry point.
        .globl  main
        .p2align  1
        .type   main,@function
main:
        ret

        .size   main, .-main

# Test R_RISCV_PCREL_HI20 and R_RISCV_PCREL_LO
# jitlink-check: decode_operand(test_pcrel32, 1) = ((named_data - test_pcrel32) + 0x800)[31:12]
# jitlink-check: decode_operand(test_pcrel32+4, 2)[11:0] = (named_data - test_pcrel32)[11:0]
        .globl  test_pcrel32
        .p2align  1
        .type   test_pcrel32,@function
test_pcrel32:
        auipc a0, %pcrel_hi(named_data)
        lw  a0, %pcrel_lo(test_pcrel32)(a0)

        .size   test_pcrel32, .-test_pcrel32

        .data
        .type   named_data,@object
        .p2align  1
named_data:
        .quad   42
        .size   named_data, 4
