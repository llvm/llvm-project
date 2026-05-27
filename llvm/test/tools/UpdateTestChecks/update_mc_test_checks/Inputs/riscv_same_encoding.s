// RUN: llvm-mc -triple=riscv32 -show-encoding < %s | FileCheck %s

addi a0, a0, 12
