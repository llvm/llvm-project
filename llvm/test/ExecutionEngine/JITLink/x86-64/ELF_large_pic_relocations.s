# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=x86_64-unknown-linux -position-independent \
# RUN:     -filetype=obj -large-code-model -o %t/elf_lg_pic_reloc.o %s
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0xfff00000 -slab-page-size 4096 \
# RUN:     -check %s %t/elf_lg_pic_reloc.o
#
# Test ELF large/PIC relocations.

        .text
        .file   "testcase.c"

        # Empty main entry point.
        .globl  main
        .p2align        4, 0x90
        .type   main,@function
main:
        retq

        .size   main, .-main

# Test R_X86_64_GOTPC64 handling. We want to check that the offset of the
# operand is the 64-bit delta to the start of the GOT.
# jitlink-check: decode_operand(test_gotpc64, 1) = \
# jitlink-check:   _GLOBAL_OFFSET_TABLE_ - test_lg_pic_GOT
# jitlink-check: decode_operand(test_got64, 1) = \
# jitlink-check:   got_addr(elf_lg_pic_reloc.o, named_data) - \
# jitlink-check:     _GLOBAL_OFFSET_TABLE_
        .globl test_lg_pic_GOT
        .p2align    4, 0x90
        .type   test_lg_pic_GOT,@function
test_lg_pic_GOT:
.L0$pb:
        leaq    .L0$pb(%rip), %rax

        .globl test_gotpc64
test_gotpc64:
        movabsq $_GLOBAL_OFFSET_TABLE_-.L0$pb, %rcx
        .size   test_gotpc64, .-test_gotpc64

        addq    %rax, %rcx
        .globl test_got64
test_got64:
        movabsq $named_data@GOT, %rax
        .size   test_got64, .-test_got64

        .size   test_lg_pic_GOT, .-test_lg_pic_GOT

        .data

        .type   named_data,@object
        .p2align        3
named_data:
        .quad   42
        .size   named_data, 8

        .ident  "clang version 10.0.0-4ubuntu1 "
        .section        ".note.GNU-stack","",@progbits
        .addrsig
