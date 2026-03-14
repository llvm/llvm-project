// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64


//------------------------------------------------------------------------------
// Vector Integer Saturating Shift Lef (Signed)
//------------------------------------------------------------------------------
         sqshl v0.8b, v1.8b, v2.8b
         sqshl v0.16b, v1.16b, v2.16b
         sqshl v0.4h, v1.4h, v2.4h
         sqshl v0.8h, v1.8h, v2.8h
         sqshl v0.2s, v1.2s, v2.2s
         sqshl v0.4s, v1.4s, v2.4s
         sqshl v0.2d, v1.2d, v2.2d

// CHECK: sqshl v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x4c,0x22,0x0e]
// CHECK: sqshl v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x4c,0x22,0x4e]
// CHECK: sqshl v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x4c,0x62,0x0e]
// CHECK: sqshl v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x4c,0x62,0x4e]
// CHECK: sqshl v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x4c,0xa2,0x0e]
// CHECK: sqshl v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x4c,0xa2,0x4e]
// CHECK: sqshl v0.2d, v1.2d, v2.2d        // encoding: [0x20,0x4c,0xe2,0x4e]

//------------------------------------------------------------------------------
// Vector Integer Saturating Shift Lef (Unsigned)
//------------------------------------------------------------------------------
         uqshl v0.8b, v1.8b, v2.8b
         uqshl v0.16b, v1.16b, v2.16b
         uqshl v0.4h, v1.4h, v2.4h
         uqshl v0.8h, v1.8h, v2.8h
         uqshl v0.2s, v1.2s, v2.2s
         uqshl v0.4s, v1.4s, v2.4s
         uqshl v0.2d, v1.2d, v2.2d

// CHECK: uqshl v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x4c,0x22,0x2e]
// CHECK: uqshl v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x4c,0x22,0x6e]
// CHECK: uqshl v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x4c,0x62,0x2e]
// CHECK: uqshl v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x4c,0x62,0x6e]
// CHECK: uqshl v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x4c,0xa2,0x2e]
// CHECK: uqshl v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x4c,0xa2,0x6e]
// CHECK: uqshl v0.2d, v1.2d, v2.2d        // encoding: [0x20,0x4c,0xe2,0x6e]

