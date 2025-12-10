# RUN: llvm-mc -triple=i386-unknown-linux-gnu -filetype=obj --show-encoding --show-inst -o %t.o %s
# RUN: llvm-jitlink -noexec \
# RUN:   -slab-allocate 1Kb -slab-address 0x1 -slab-page-size 4096 \
# RUN:   -abs external_data=0x32 \
# RUN:   -check %s %t.o
#
# Test ELF 16 bit absolute relocations

        .text
        .code16 

        .globl  main     
        .align        2, 0x90
        .type   main,@function
main:      
        ret
        .size   main, .-main

# jitlink-check: decode_operand(bar, 0) = external_data
        .globl  bar
        .align        2, 0x90
        .type   bar,@function
bar:
        retw    $external_data
        .size   bar, .-bar

# jitlink-check: decode_operand(baz, 0) = external_data + 23
        .globl  baz
        .align        2, 0x90
        .type   baz,@function
baz:
        retw    $external_data+23
        .size   baz, .-baz
