// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64


//------------------------------------------------------------------------------
// Vector Integer Rounding Shift Lef (Signed)
//------------------------------------------------------------------------------
         srshl v0.8b, v1.8b, v2.8b
         srshl v0.16b, v1.16b, v2.16b
         srshl v0.4h, v1.4h, v2.4h
         srshl v0.8h, v1.8h, v2.8h
         srshl v0.2s, v1.2s, v2.2s
         srshl v0.4s, v1.4s, v2.4s
         srshl v0.2d, v1.2d, v2.2d

// CHECK: srshl v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x54,0x22,0x0e]
// CHECK: srshl v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x54,0x22,0x4e]
// CHECK: srshl v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x54,0x62,0x0e]
// CHECK: srshl v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x54,0x62,0x4e]
// CHECK: srshl v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x54,0xa2,0x0e]
// CHECK: srshl v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x54,0xa2,0x4e]
// CHECK: srshl v0.2d, v1.2d, v2.2d        // encoding: [0x20,0x54,0xe2,0x4e]

//------------------------------------------------------------------------------
// Vector Integer Rounding Shift Lef (Unsigned)
//------------------------------------------------------------------------------
         urshl v0.8b, v1.8b, v2.8b
         urshl v0.16b, v1.16b, v2.16b
         urshl v0.4h, v1.4h, v2.4h
         urshl v0.8h, v1.8h, v2.8h
         urshl v0.2s, v1.2s, v2.2s
         urshl v0.4s, v1.4s, v2.4s
         urshl v0.2d, v1.2d, v2.2d

// CHECK: urshl v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x54,0x22,0x2e]
// CHECK: urshl v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x54,0x22,0x6e]
// CHECK: urshl v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x54,0x62,0x2e]
// CHECK: urshl v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x54,0x62,0x6e]
// CHECK: urshl v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x54,0xa2,0x2e]
// CHECK: urshl v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x54,0xa2,0x6e]
// CHECK: urshl v0.2d, v1.2d, v2.2d        // encoding: [0x20,0x54,0xe2,0x6e]



