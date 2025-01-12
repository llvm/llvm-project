// RUN: llvm-mc -triple aarch64 -o - %s 2>&1 | FileCheck %s

.arch armv9-a+sme2p2
ftmopa  za0.s, {z0.s-z1.s}, z0.s, z20[0]
// CHECK: ftmopa za0.s, { z0.s, z1.s }, z0.s, z20[0]
