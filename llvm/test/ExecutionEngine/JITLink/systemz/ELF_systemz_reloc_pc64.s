# REQUIRES: system-linux
# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=systemz-unknown-linux -position-independent \
# RUN:     -filetype=obj -o  %t/elf_reloc.o %s
#
# RUN: llvm-jitlink -noexec \
# RUN:     -slab-allocate 100Kb -slab-address 0xffff0000 -slab-page-size 4096 \
# RUN:     -abs external_data=0x1 \
# RUN:    -abs foo=0x6ff04040 \
# RUN:    -abs bar=0x6ff04048 \
# RUN:     -check %s %t/elf_reloc.o

        .text
        .section        .text.main
        .globl  main
        .p2align        4
        .type   main,@function
main:
        br    %r14
        .size   main, .-main

        .globl test_pc64_foo
# jitlink-check: *{8}test_pc64_foo = foo - test_pc64_foo
test_pc64_foo:
        .reloc ., R_390_PC64, foo
        .space 8
        .size test_pc64_foo, .-test_pc64_foo

        .globl test_pc64_bar
# jitlink-check: *{8}test_pc64_bar = bar - test_pc64_bar
test_pc64_bar:
        .reloc ., R_390_PC64, bar 
        .space 8
        .size test_pc64_bar, .-test_pc64_bar
