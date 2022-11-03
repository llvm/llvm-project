# RUN: llvm-mc -triple=i386-unknown-linux-gnu -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec %t.o

        .text
        .globl  main
        .p2align        4
        .type   main,@function
main:
    pushl   %ebp
    movl    %esp, %ebp
    pushl   %eax
    movl    $0, -4(%ebp)
    movl    a, %eax
    addl    $4, %esp
    popl    %ebp
    retl

        .size   main, .-main

        .data
        .p2align        2
        .type   a,@object
a:
        .long   42
        .size   a, 4