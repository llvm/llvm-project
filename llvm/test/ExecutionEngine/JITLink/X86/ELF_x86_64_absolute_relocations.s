# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-unknown-linux -position-independent \
# RUN:     -filetype=obj -o %t/elf_abs_reloc.o %s
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0xfff00000 -slab-page-size 4096 \
# RUN:     -abs external_data_low=0x1 \
# RUN:     -abs external_data_high=0xffffffff80000000 \
# RUN:     -check %s %t/elf_abs_reloc.o
#
# Test ELF absolute relocations.


        .text
        .file   "testcase.c"

# Empty main entry point.
        .globl  main
        .p2align        4, 0x90
        .type   main,@function
main:
        retq

        .size   main, .-main

# R_X86_64_32S handling
# Test the target value is in range of signed 32-bits imm
# jitlink-check: decode_operand(test_abs_32S, 4) = external_data_low
# jitlink-check: decode_operand(test_abs_32S+7, 4)[31:0] = external_data_high[31:0]
        .globl  test_abs_32S
        .p2align       4, 0x90
        .type   test_abs_32S,@function
test_abs_32S:
        movl    external_data_low, %eax
        movl    external_data_high, %esi

         .size   test_abs_32S, .-test_abs_32S
