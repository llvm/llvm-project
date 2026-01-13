# RUN: llvm-mc -triple=aarch64-unknown-linux-gnu \
# RUN:   -position-independent -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -show-addrs %t.o | FileCheck %s

# Check that show-addrs works with zero-sized symbols.

# CHECK: main{{.*}}target addr =

        .text
        .globl  main
        .p2align        2
        .type   main,@function
main:
        mov     w0, wzr
        ret
        .size   main, 0
