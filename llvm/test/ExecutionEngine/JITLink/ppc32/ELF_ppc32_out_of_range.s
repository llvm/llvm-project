# RUN: llvm-mc -triple=powerpc-unknown-linux-gnu -filetype=obj %s -o %t.o
# RUN: not llvm-jitlink -noexec -abs big=0x12345 %t.o 2>&1 | FileCheck %s

# R_PPC_ADDR16 overflow.
# CHECK: out of range of Pointer16 fixup

        .text
        .globl main
main:
        blr

        .data
        .globl data
data:
        .short big
