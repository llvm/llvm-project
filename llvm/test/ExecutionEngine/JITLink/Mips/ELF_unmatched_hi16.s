# RUN: llvm-mc -triple=mipsel-unknown-linux-gnu -mcpu=mips32r2 -filetype=obj %s -o %t.o
# RUN: not llvm-jitlink -noexec %t.o 2>&1 | FileCheck %s

        .text
        .globl main
main:
        lui $2, %hi(target)
        jr $ra
        nop

        .data
target:
        .word 0

# CHECK: unmatched R_MIPS_HI16 relocation
