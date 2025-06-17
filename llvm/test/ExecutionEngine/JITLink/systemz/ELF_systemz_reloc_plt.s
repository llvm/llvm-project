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

# Check R_390_PLT32/64 relocations.

        .text
        .section        .text.main
        .globl  main
        .p2align        4
        .type   main,@function
main:
        br   %r14
        .size main, .-main

        .globl test_plt32_foo
# jitlink-check: *{4}test_plt32_foo = \
# jitlink-check:  stub_addr(elf_reloc.o, foo) - test_plt32_foo
test_plt32_foo:
        .reloc ., R_390_PLT32, foo
        .space 4
        .size test_plt32_foo, .-test_plt32_foo

        .globl test_plt32_bar
# jitlink-check: *{4}test_plt32_bar = \
# jitlink-check:  stub_addr(elf_reloc.o, bar) - test_plt32_bar
test_plt32_bar:
        .reloc ., R_390_PLT32, bar
        .space 4
        .size test_plt32_bar, .-test_plt32_bar

       .globl test_plt64_foo
# jitlink-check: *{8}test_plt64_foo = \
# jitlink-check:  stub_addr(elf_reloc.o, foo) - test_plt64_foo
test_plt64_foo:
        .reloc ., R_390_PLT64, foo
        .space 8
        .size test_plt64_foo, .-test_plt64_foo

       .globl test_plt64_bar
# jitlink-check: *{8}test_plt64_bar = \
# jitlink-check:  stub_addr(elf_reloc.o, bar) - test_plt64_bar
test_plt64_bar:
        .reloc ., R_390_PLT64, bar
        .space 8
        .size test_plt64_bar, .-test_plt64_bar

        .globl test_plt32dbl_foo
# jitlink-check: *{4}test_plt32dbl_foo = \
# jitlink-check:  (stub_addr(elf_reloc.o, foo) - test_plt32dbl_foo) >> 1
test_plt32dbl_foo:
        .reloc ., R_390_PLT32DBL, foo
        .space 4
        .size test_plt32dbl_foo, .-test_plt32dbl_foo

        .globl test_plt32dbl_bar
# jitlink-check: *{4}test_plt32dbl_bar = \
# jitlink-check:  (stub_addr(elf_reloc.o, bar) - test_plt32dbl_bar) >> 1
test_plt32dbl_bar:
        .reloc ., R_390_PLT32DBL, bar
        .space 4
        .size test_plt32dbl_bar, .-test_plt32dbl_bar

