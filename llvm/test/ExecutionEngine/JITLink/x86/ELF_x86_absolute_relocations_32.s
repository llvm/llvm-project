# RUN: llvm-mc -triple=i386-unknown-linux-gnu -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec \
# RUN:   -slab-allocate 100Kb -slab-address 0xfff00000 -slab-page-size 4096 \
# RUN:   -abs external_data=0x100 \
# RUN:   -check %s %t.o

# Test ELF 32 bit absolute relocations

        .text
        .globl  main
        .p2align        4, 0x90
        .type   main,@function
main:
        retl
        .size   main, .-main

# jitlink-check: decode_operand(foo, 0) = external_data
        .globl  foo
        .p2align        4, 0x90
        .type   foo,@function
foo:
        movl    external_data, %eax
        .size   foo, .-foo

# jitlink-check: decode_operand(bar, 0) = external_data + 4000
        .globl  bar
        .p2align        4, 0x90
        .type   bar,@function
bar:
        movl    external_data + 4000, %eax
        .size   bar, .-bar
