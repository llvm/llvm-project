# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=mipsel-unknown-linux-gnu -mcpu=mips1 -filetype=obj %s -o %t/o32el.o
# RUN: llvm-mc -triple=mips-unknown-linux-gnu -mcpu=mips1 -filetype=obj %s -o %t/o32be.o
# RUN: llvm-mc -triple=mips64el-unknown-linux-gnuabi64 -mcpu=mips64 -filetype=obj %s -o %t/n64el.o
# RUN: llvm-mc -triple=mips64-unknown-linux-gnuabi64 -mcpu=mips64 -filetype=obj %s -o %t/n64be.o
# RUN: llvm-jitlink -noexec --abs=small_abs=0x1234 -check=%s %t/o32el.o
# RUN: llvm-jitlink -noexec --abs=small_abs=0x1234 -check=%s %t/o32be.o
# RUN: llvm-jitlink -noexec --abs=small_abs=0x1234 -check=%s %t/n64el.o
# RUN: llvm-jitlink -noexec --abs=small_abs=0x1234 -check=%s %t/n64be.o

        .set noreorder
        .text
        .globl main
        .type main,@function
main:
        jr $ra
        nop
        .size main, .-main

        .data
        .globl half_pointer
half_pointer:
        .2byte small_abs
        .2byte 0xbeef

# jitlink-check: *{2}half_pointer = 0x1234
# jitlink-check: *{2}(half_pointer + 2) = 0xbeef
