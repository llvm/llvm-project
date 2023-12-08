// RUN: llvm-mc -triple=aarch64-none-linux-gnu -mattr=+sve -show-encoding < %s 2>&1 | FileCheck %s

foo:
// CHECK-NOT: error:
  pbar .req p1

// CHECK: add z0.s, z1.s, z2.s
  zbar .req z1
  add  z0.s, zbar.s, z2.s
