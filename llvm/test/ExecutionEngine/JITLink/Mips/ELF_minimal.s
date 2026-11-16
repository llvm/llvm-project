# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=mipsel-unknown-linux-gnu -mcpu=mips1 -filetype=obj %s -o %t/mips1el.o
# RUN: llvm-mc -triple=mips-unknown-linux-gnu -mcpu=mips1 -filetype=obj %s -o %t/mips1be.o
# RUN: llvm-mc -triple=mips64el-unknown-linux-gnuabi64 -mcpu=mips64 -filetype=obj %s -o %t/mips64el.o
# RUN: llvm-mc -triple=mips64-unknown-linux-gnuabi64 -mcpu=mips64 -filetype=obj %s -o %t/mips64be.o
# RUN: llvm-jitlink -noexec -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 %t/mips1el.o
# RUN: llvm-jitlink -noexec -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 %t/mips1be.o
# RUN: llvm-jitlink -noexec -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 %t/mips64el.o
# RUN: llvm-jitlink -noexec -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 %t/mips64be.o

        .set noreorder
        .text
        .globl main
        .type main,@function
main:
        lui $2, %hi(data)
        addiu $2, $2, %lo(data)
        jr $ra
        nop
        .size main, .-main

        .data
data:
        .word 42
