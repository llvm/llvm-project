// This test verifies SME2 implies SME.

// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sme2  2>&1 < %s \
// RUN:        | FileCheck %s

addha   za0.s, p0/m, p0/m, z0.s
// CHECK-NOT: instruction requires: sme
// CHECK: addha   za0.s, p0/m, p0/m, z0.s
