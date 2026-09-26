# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=mips64el-unknown-linux-gnuabi64 -mcpu=mips64 -filetype=obj %s -o %t/le.o
# RUN: llvm-mc -triple=mips64-unknown-linux-gnuabi64 -mcpu=mips64 -filetype=obj %s -o %t/be.o
# RUN: llvm-jitlink -noexec -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 -check=%s %t/le.o
# RUN: llvm-jitlink -noexec -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 -check=%s %t/be.o

        .set noreorder
        .text
        .globl main
        .type main,@function
main:
        jr $ra
        nop
        .size main, .-main

        .data
        .p2align 3
        .globl target
target:
        .8byte 0

        .globl gp_value
gp_value:
        .gpdword target

        .globl delta_value
delta_value:
        .8byte target - .

# jitlink-check: *{8}gp_value = target - _gp
# jitlink-check: *{8}delta_value = target - delta_value
