# RUN: llvm-mc -triple=mipsel-unknown-linux-gnu -mcpu=mips32r2 -filetype=obj %s -o %t.o
# RUN: not llvm-jitlink -noexec --abs=far_target=0x30000000 -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 %t.o 2>&1 | FileCheck %s --check-prefix=RANGE
# RUN: not llvm-jitlink -noexec --abs=far_target=0x10008002 -slab-address=0x10000000 -slab-allocate=128Kb -slab-page-size=4096 %t.o 2>&1 | FileCheck %s --check-prefix=ALIGN

        .set noreorder
        .text
        .globl main
main:
        beq $zero, $zero, far_target
        nop

# RANGE: is out of range of PC16 fixup
# ALIGN: improper alignment for relocation
