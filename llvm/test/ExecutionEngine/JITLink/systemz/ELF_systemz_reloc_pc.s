# REQUIRES: system-linux
# RUN: llvm-mc -triple=systemz-unknown-linux -position-independent \
# RUN:     -filetype=obj -o %t.o %s
#
# RUN: llvm-jitlink -noexec %t.o
#
# Check R_390_PC* handling.

        .text
        .globl  main
        .type   main,@function
main:
        br      %r14 
        .size   main, .-main

        .rodata
        .short  main-. # Generate R_390_PC16 relocation.
        .long   main-. # Generate R_390_PC32 relocation.
        .quad   main-. # Generate R_390_PC64 relocation.

