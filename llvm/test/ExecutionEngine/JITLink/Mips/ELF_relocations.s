# RUN: rm -rf %t && mkdir -p %t
# RUN: llvm-mc -triple=mipsel-unknown-linux-gnu -mcpu=mips32r2 -filetype=obj %s -o %t/o32el.o
# RUN: llvm-mc -triple=mips-unknown-linux-gnu -mcpu=mips32r2 -filetype=obj %s -o %t/o32be.o
# RUN: llvm-mc -triple=mips64el-unknown-linux-gnuabin32 -mcpu=mips64r2 -filetype=obj %s -o %t/n32el.o
# RUN: llvm-mc -triple=mips64-unknown-linux-gnuabin32 -mcpu=mips64r2 -filetype=obj %s -o %t/n32be.o
# RUN: llvm-mc -triple=mips64el-unknown-linux-gnuabi64 -mcpu=mips64r2 -defsym PTR64=1 -filetype=obj %s -o %t/n64el.o
# RUN: llvm-mc -triple=mips64-unknown-linux-gnuabi64 -mcpu=mips64r2 -defsym PTR64=1 -filetype=obj %s -o %t/n64be.o
# RUN: llvm-jitlink -noexec -slab-address 0x10000000 -slab-allocate 128Kb -slab-page-size 4096 -check %s -check-name=jitlink-check-32 %t/o32el.o
# RUN: llvm-jitlink -noexec -slab-address 0x10000000 -slab-allocate 128Kb -slab-page-size 4096 -check %s -check-name=jitlink-check-32 %t/o32be.o
# RUN: llvm-jitlink -noexec -slab-address 0x10000000 -slab-allocate 128Kb -slab-page-size 4096 -check %s -check-name=jitlink-check-32 %t/n32el.o
# RUN: llvm-jitlink -noexec -slab-address 0x10000000 -slab-allocate 128Kb -slab-page-size 4096 -check %s -check-name=jitlink-check-32 %t/n32be.o
# RUN: llvm-jitlink -noexec -slab-address 0x10000000 -slab-allocate 128Kb -slab-page-size 4096 -check %s -check-name=jitlink-check-64 %t/n64el.o
# RUN: llvm-jitlink -noexec -slab-address 0x10000000 -slab-allocate 128Kb -slab-page-size 4096 -check %s -check-name=jitlink-check-64 %t/n64be.o

        .set noreorder
        .text
        .globl main
        .type main,@function
main:
        .cfi_startproc
        jr $ra
        nop
        .cfi_endproc
        .size main, .-main

# jitlink-check-32: decode_operand(load_address, 1) = (target + 0x8008)[31:16]
# jitlink-check-32: decode_operand(load_address + 4, 2)[15:0] = (target + 8)[15:0]
# jitlink-check-64: decode_operand(load_address, 1) = (target + 0x8008)[31:16]
# jitlink-check-64: decode_operand(load_address + 4, 2)[15:0] = (target + 8)[15:0]
        .globl load_address
        .type load_address,@function
load_address:
        lui $4, %hi(target + 8)
        addiu $4, $4, %lo(target + 8)
        jr $ra
        nop
        .size load_address, .-load_address

        .data
        .p2align 3
        .globl target
target:
        .8byte 0x1122334455667788

        .globl target_pointer
target_pointer:
        .ifdef PTR64
        .8byte target + 4
        .else
        .4byte target + 4
        .endif

# jitlink-check-32: *{4}(target_pointer) = target + 4
# jitlink-check-64: *{8}(target_pointer) = target + 4
