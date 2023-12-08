// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.5a < %s | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.4a,+fptoint < %s | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.4a,-v8.5a < %s 2>&1 | FileCheck %s --check-prefix=NOFRINT
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=-v8.5a < %s 2>&1 | FileCheck %s --check-prefix=NOFRINT

// FP-to-int rounding, scalar
frint32z s0, s1
frint32z d0, d1
frint64z s2, s3
frint64z d2, d3
frint32x s4, s5
frint32x d4, d5
frint64x s6, s7
frint64x d6, d7

// CHECK: frint32z s0, s1                 // encoding: [0x20,0x40,0x28,0x1e]
// CHECK: frint32z d0, d1                 // encoding: [0x20,0x40,0x68,0x1e]
// CHECK: frint64z s2, s3                 // encoding: [0x62,0x40,0x29,0x1e]
// CHECK: frint64z d2, d3                 // encoding: [0x62,0x40,0x69,0x1e]
// CHECK: frint32x s4, s5                 // encoding: [0xa4,0xc0,0x28,0x1e]
// CHECK: frint32x d4, d5                 // encoding: [0xa4,0xc0,0x68,0x1e]
// CHECK: frint64x s6, s7                 // encoding: [0xe6,0xc0,0x29,0x1e]
// CHECK: frint64x d6, d7                 // encoding: [0xe6,0xc0,0x69,0x1e]

// NOFRINT: instruction requires: frint3264
// NOFRINT-NEXT: frint32z s0, s1
// NOFRINT: instruction requires: frint3264
// NOFRINT-NEXT: frint32z d0, d1
// NOFRINT: instruction requires: frint3264
// NOFRINT-NEXT: frint64z s2, s3
// NOFRINT: instruction requires: frint3264
// NOFRINT-NEXT: frint64z d2, d3
// NOFRINT: instruction requires: frint3264
// NOFRINT-NEXT: frint32x s4, s5
// NOFRINT: instruction requires: frint3264
// NOFRINT-NEXT: frint32x d4, d5
// NOFRINT: instruction requires: frint3264
// NOFRINT-NEXT: frint64x s6, s7
// NOFRINT: instruction requires: frint3264
// NOFRINT-NEXT: frint64x d6, d7

// FP-to-int rounding, vector
frint32z v0.2s, v1.2s
frint32z v0.2d, v1.2d
frint32z v0.4s, v1.4s
frint64z v2.2s, v3.2s
frint64z v2.2d, v3.2d
frint64z v2.4s, v3.4s
frint32x v4.2s, v5.2s
frint32x v4.2d, v5.2d
frint32x v4.4s, v5.4s
frint64x v6.2s, v7.2s
frint64x v6.2d, v7.2d
frint64x v6.4s, v7.4s

// CHECK: frint32z v0.2s, v1.2s           // encoding: [0x20,0xe8,0x21,0x0e]
// CHECK: frint32z v0.2d, v1.2d           // encoding: [0x20,0xe8,0x61,0x4e]
// CHECK: frint32z v0.4s, v1.4s           // encoding: [0x20,0xe8,0x21,0x4e]
// CHECK: frint64z v2.2s, v3.2s           // encoding: [0x62,0xf8,0x21,0x0e]
// CHECK: frint64z v2.2d, v3.2d           // encoding: [0x62,0xf8,0x61,0x4e]
// CHECK: frint64z v2.4s, v3.4s           // encoding: [0x62,0xf8,0x21,0x4e]
// CHECK: frint32x v4.2s, v5.2s           // encoding: [0xa4,0xe8,0x21,0x2e]
// CHECK: frint32x v4.2d, v5.2d           // encoding: [0xa4,0xe8,0x61,0x6e]
// CHECK: frint32x v4.4s, v5.4s           // encoding: [0xa4,0xe8,0x21,0x6e]
// CHECK: frint64x v6.2s, v7.2s           // encoding: [0xe6,0xf8,0x21,0x2e]
// CHECK: frint64x v6.2d, v7.2d           // encoding: [0xe6,0xf8,0x61,0x6e]
// CHECK: frint64x v6.4s, v7.4s           // encoding: [0xe6,0xf8,0x21,0x6e]

// NOFRINT: instruction requires: frint3264
// NOFRINT-NEXT: frint32z v0.2s, v1.2s
// NOFRINT: instruction requires: frint3264
// NOFRINT-NEXT: frint32z v0.2d, v1.2d
// NOFRINT: instruction requires: frint3264
// NOFRINT-NEXT: frint32z v0.4s, v1.4s
// NOFRINT: instruction requires: frint3264
// NOFRINT-NEXT: frint64z v2.2s, v3.2s
// NOFRINT: instruction requires: frint3264
// NOFRINT-NEXT: frint64z v2.2d, v3.2d
// NOFRINT: instruction requires: frint3264
// NOFRINT-NEXT: frint64z v2.4s, v3.4s
// NOFRINT: instruction requires: frint3264
// NOFRINT-NEXT: frint32x v4.2s, v5.2s
// NOFRINT: instruction requires: frint3264
// NOFRINT-NEXT: frint32x v4.2d, v5.2d
// NOFRINT: instruction requires: frint3264
// NOFRINT-NEXT: frint32x v4.4s, v5.4s
// NOFRINT: instruction requires: frint3264
// NOFRINT-NEXT: frint64x v6.2s, v7.2s
// NOFRINT: instruction requires: frint3264
// NOFRINT-NEXT: frint64x v6.2d, v7.2d
// NOFRINT: instruction requires: frint3264
// NOFRINT-NEXT: frint64x v6.4s, v7.4s
