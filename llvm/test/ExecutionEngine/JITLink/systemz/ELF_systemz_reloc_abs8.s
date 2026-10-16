# REQUIRES: system-linux
# RUN: llvm-mc -triple=systemz-unknown-linux -position-independent \
# RUN:     -filetype=obj -o %t.o %s
# RUN: llvm-jitlink -noexec -abs X=0xFF -check=%s %t.o

# RUN: not llvm-jitlink -noexec -abs X=0x100 %t.o 2>&1 | \
# RUN:   FileCheck -check-prefix=CHECK-ERROR %s
#
# Check success and failure cases of R_390_8 handling.

# jitlink-check: *{8}P = X

# CHECK-ERROR: relocation target {{.*}} (X) is out of range of Pointer8 fixup

        .text
        .section        .text.main
        .globl  main
        .p2align        4
        .type   main,@function
main:
        br    %r14
.Lfunc_end0:
        .size   main, .Lfunc_end0-main

        .type   P,@object
        .data
        .globl  P
P:
        .byte   0
        .byte   0
        .byte   0
        .byte   0
        .byte   0
        .byte   0
        .byte   0
        .byte   X    # Using byte here generates R_390_8.
        .size   P, 8

