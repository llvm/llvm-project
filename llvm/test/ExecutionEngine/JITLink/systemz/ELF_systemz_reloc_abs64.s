# REQUIRES: system-linux
# RUN: llvm-mc -triple=systemz-unknown-linux -position-independent \
# RUN:     -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -abs X=0xffffffffffffffff -check=%s %t.o
#
# Check success and failure cases of R_390_64 handling.

# jitlink-check: *{8}P = X

# CHECK-ERROR: relocation target "X" {{.*}} is out of range of Pointer64 fixup

        .text
        .section        .text.main
        .globl  main
        .p2align        4
        .type   main,@function
main:
        br  %r14
.Lfunc_end0:
        .size   main, .Lfunc_end0-main

        .type   P,@object
        .data
        .globl  P
        .p2align       4 
P:
        .quad  X    # Using quad here generates R_390_64.
        .size   P, 8
