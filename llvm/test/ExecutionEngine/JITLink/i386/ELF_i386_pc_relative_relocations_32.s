# RUN: llvm-mc -triple=i386-unknown-linux-gnu -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec %t.o

        .text
        .section        .text.main
        .globl  main
        .p2align        4
        .type   main,@function
main:
    pushl   %ebp
    movl    %esp, %ebp
    subl    $8, %esp
    movl    $0, -4(%ebp)
    calll   foo
    addl    $8, %esp
    popl    %ebp
    retl

        .size   main, .-main

        .section        .text.foo
        .p2align        4
        .type   foo,@function
foo:
    pushl   %ebp
    movl    %esp, %ebp
    movl    $42, %eax
    popl    %ebp
    retl

    .size       foo, .-foo