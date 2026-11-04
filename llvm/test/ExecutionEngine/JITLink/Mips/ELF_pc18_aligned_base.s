# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=mips64el-unknown-linux-gnuabi64 -mcpu=mips64r6 -filetype=obj %s -o %t/le.o
# RUN: llvm-mc -triple=mips64-unknown-linux-gnuabi64 -mcpu=mips64r6 -filetype=obj %s -o %t/be.o
# RUN: llvm-jitlink -noexec -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 -check=%s %t/le.o
# RUN: llvm-jitlink -noexec -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 -check=%s %t/be.o

        .set noreorder
        .text
        .globl main
        .type main,@function
main:
        nop
        .globl load_pc
load_pc:
        ldpc $2, target
        jrc $ra
        .size main, .-main

        .section .rodata,"a",@progbits
        .p2align 3
target:
        .8byte 42

# R_MIPS_PC18_S3 uses the containing 8-byte bundle as its PC base.
# jitlink-check: (*{4}load_pc) & 0x3ffff = ((target - (load_pc & 0xfffffffffffffff8)) >> 3) & 0x3ffff
