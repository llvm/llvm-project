# RUN: llvm-mc -triple=mipsel-unknown-linux-gnu -mcpu=mips32r2 -mattr=+micromips -filetype=obj %s -o %t.o
# RUN: not llvm-jitlink -noexec %t.o 2>&1 | FileCheck %s
# RUN: llvm-mc -triple=mipsel-unknown-linux-gnu -mcpu=mips32r2 -mattr=+mips16 -filetype=obj %s -o %t.mips16.o
# RUN: not llvm-jitlink -noexec %t.mips16.o 2>&1 | FileCheck %s

        .set noreorder
        .text
        .globl main
main:
        nop

# CHECK: MIPS16 and microMIPS{{.*}}unsupported by JITLink
