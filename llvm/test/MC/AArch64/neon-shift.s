// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64


//------------------------------------------------------------------------------
// Vector Integer Shift Lef (Signed)
//------------------------------------------------------------------------------
         sshl v0.8b, v1.8b, v2.8b
         sshl v0.16b, v1.16b, v2.16b
         sshl v0.4h, v1.4h, v2.4h
         sshl v0.8h, v1.8h, v2.8h
         sshl v0.2s, v1.2s, v2.2s
         sshl v0.4s, v1.4s, v2.4s
         sshl v0.2d, v1.2d, v2.2d

// CHECK: sshl v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x44,0x22,0x0e]
// CHECK: sshl v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x44,0x22,0x4e]
// CHECK: sshl v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x44,0x62,0x0e]
// CHECK: sshl v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x44,0x62,0x4e]
// CHECK: sshl v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x44,0xa2,0x0e]
// CHECK: sshl v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x44,0xa2,0x4e]
// CHECK: sshl v0.2d, v1.2d, v2.2d        // encoding: [0x20,0x44,0xe2,0x4e]

//------------------------------------------------------------------------------
// Vector Integer Shift Lef (Unsigned)
//------------------------------------------------------------------------------
         ushl v0.8b, v1.8b, v2.8b
         ushl v0.16b, v1.16b, v2.16b
         ushl v0.4h, v1.4h, v2.4h
         ushl v0.8h, v1.8h, v2.8h
         ushl v0.2s, v1.2s, v2.2s
         ushl v0.4s, v1.4s, v2.4s
         ushl v0.2d, v1.2d, v2.2d

// CHECK: ushl v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x44,0x22,0x2e]
// CHECK: ushl v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x44,0x22,0x6e]
// CHECK: ushl v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x44,0x62,0x2e]
// CHECK: ushl v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x44,0x62,0x6e]
// CHECK: ushl v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x44,0xa2,0x2e]
// CHECK: ushl v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x44,0xa2,0x6e]
// CHECK: ushl v0.2d, v1.2d, v2.2d        // encoding: [0x20,0x44,0xe2,0x6e]

//------------------------------------------------------------------------------
// Vector Integer Shift Left by Immediate
//------------------------------------------------------------------------------
         shl v0.8b, v1.8b, #3
         shl v0.4h, v1.4h, #3
         shl v0.2s, v1.2s, #3
         shl v0.16b, v1.16b, #3
         shl v0.8h, v1.8h, #3
         shl v0.4s, v1.4s, #3
         shl v0.2d, v1.2d, #3

// CHECK: shl v0.8b, v1.8b, #3        // encoding: [0x20,0x54,0x0b,0x0f]
// CHECK: shl v0.4h, v1.4h, #3        // encoding: [0x20,0x54,0x13,0x0f]
// CHECK: shl v0.2s, v1.2s, #3        // encoding: [0x20,0x54,0x23,0x0f]
// CHECK: shl v0.16b, v1.16b, #3      // encoding: [0x20,0x54,0x0b,0x4f]
// CHECK: shl v0.8h, v1.8h, #3        // encoding: [0x20,0x54,0x13,0x4f]
// CHECK: shl v0.4s, v1.4s, #3        // encoding: [0x20,0x54,0x23,0x4f]
// CHECK: shl v0.2d, v1.2d, #3        // encoding: [0x20,0x54,0x43,0x4f]
