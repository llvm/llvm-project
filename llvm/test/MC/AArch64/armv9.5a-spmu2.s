// RUN: llvm-mc -triple aarch64 -show-encoding < %s | FileCheck %s

msr SPMZR_EL0, x0
// CHECK: msr SPMZR_EL0, x0                  // encoding: [0x80,0x9c,0x13,0xd5]
