// This test verifies SVE2p1 implies SVE2.

// RUN: llvm-mc -triple=aarch64 -show-encoding -mattr=+sve2p1  2>&1 < %s \
// RUN:        | FileCheck %s

cmla   z0.b, z1.b, z2.b, #0
// CHECK: cmla   z0.b, z1.b, z2.b, #0
