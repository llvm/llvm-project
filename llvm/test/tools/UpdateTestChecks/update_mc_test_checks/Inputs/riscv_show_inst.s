// RUN: llvm-mc -triple=riscv32 -mattr=+experimental-y -show-encoding -show-inst %s | FileCheck %s
add a0, a1, a2
