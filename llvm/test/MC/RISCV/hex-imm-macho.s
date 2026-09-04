// RUN: llvm-mc -triple riscv32-apple-unknown-macho -mattr=+c --riscv-no-aliases %s | FileCheck %s

// CHECK: andi a0, a0, 0x190
// CHECK: ori a0, a0, 0x400
// CHECK: xori a0, a0, 0xfffffff4
andi a0, a0, 400
ori a0, a0, 1024
xori a0, a0, -12

// CHECK: c.andi s0, 0x1f
c.andi s0, 31

// CHECK: auipc a0, 0x3e8
// CHECK: lui a0, 0x2710
auipc a0, 1000
lui a0, 10000
