// RUN: llvm-mc -triple arm64 -mattr=crypto -show-encoding < %s | FileCheck %s

pmull v8.8h, v8.8b, v8.8b
pmull2 v8.8h, v8.16b, v8.16b
pmull v8.1q, v8.1d, v8.1d
pmull2 v8.1q, v8.2d, v8.2d
// CHECK: pmull v8.8h, v8.8b, v8.8b    // encoding: [0x08,0xe1,0x28,0x0e]
// CHECK: pmull2 v8.8h, v8.16b, v8.16b // encoding: [0x08,0xe1,0x28,0x4e]
// CHECK: pmull v8.1q, v8.1d, v8.1d    // encoding: [0x08,0xe1,0xe8,0x0e]
// CHECK: pmull2 v8.1q, v8.2d, v8.2d   // encoding: [0x08,0xe1,0xe8,0x4e]

pmull v8.8H, v8.8B, v8.8B
pmull2 v8.8H, v8.16B, v8.16B
pmull v8.1Q, v8.1D, v8.1D
pmull2 v8.1Q, v8.2D, v8.2D
// CHECK: pmull v8.8h, v8.8b, v8.8b    // encoding: [0x08,0xe1,0x28,0x0e]
// CHECK: pmull2 v8.8h, v8.16b, v8.16b // encoding: [0x08,0xe1,0x28,0x4e]
// CHECK: pmull v8.1q, v8.1d, v8.1d    // encoding: [0x08,0xe1,0xe8,0x0e]
// CHECK: pmull2 v8.1q, v8.2d, v8.2d   // encoding: [0x08,0xe1,0xe8,0x4e]
