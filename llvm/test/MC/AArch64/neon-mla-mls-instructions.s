// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//----------------------------------------------------------------------
// Vector Integer Multiply-accumulate
//----------------------------------------------------------------------
         mla v0.8b, v1.8b, v2.8b
         mla v0.16b, v1.16b, v2.16b
         mla v0.4h, v1.4h, v2.4h
         mla v0.8h, v1.8h, v2.8h
         mla v0.2s, v1.2s, v2.2s
         mla v0.4s, v1.4s, v2.4s

// CHECK: mla v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x94,0x22,0x0e]
// CHECK: mla v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x94,0x22,0x4e]
// CHECK: mla v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x94,0x62,0x0e]
// CHECK: mla v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x94,0x62,0x4e]
// CHECK: mla v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x94,0xa2,0x0e]
// CHECK: mla v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x94,0xa2,0x4e]


//----------------------------------------------------------------------
// Vector Integer Multiply-subtract
//----------------------------------------------------------------------
         mls v0.8b, v1.8b, v2.8b
         mls v0.16b, v1.16b, v2.16b
         mls v0.4h, v1.4h, v2.4h
         mls v0.8h, v1.8h, v2.8h
         mls v0.2s, v1.2s, v2.2s
         mls v0.4s, v1.4s, v2.4s

// CHECK: mls v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x94,0x22,0x2e]
// CHECK: mls v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x94,0x22,0x6e]
// CHECK: mls v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x94,0x62,0x2e]
// CHECK: mls v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x94,0x62,0x6e]
// CHECK: mls v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x94,0xa2,0x2e]
// CHECK: mls v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x94,0xa2,0x6e]

//----------------------------------------------------------------------
// Vector Floating-Point Multiply-accumulate
//----------------------------------------------------------------------
         fmla v0.2s, v1.2s, v2.2s
         fmla v0.4s, v1.4s, v2.4s
         fmla v0.2d, v1.2d, v2.2d

// CHECK: fmla v0.2s, v1.2s, v2.2s       // encoding: [0x20,0xcc,0x22,0x0e]
// CHECK: fmla v0.4s, v1.4s, v2.4s       // encoding: [0x20,0xcc,0x22,0x4e]
// CHECK: fmla v0.2d, v1.2d, v2.2d       // encoding: [0x20,0xcc,0x62,0x4e]

//----------------------------------------------------------------------
// Vector Floating-Point Multiply-subtract
//----------------------------------------------------------------------
         fmls v0.2s, v1.2s, v2.2s
         fmls v0.4s, v1.4s, v2.4s
         fmls v0.2d, v1.2d, v2.2d

// CHECK: fmls v0.2s, v1.2s, v2.2s       // encoding: [0x20,0xcc,0xa2,0x0e]
// CHECK: fmls v0.4s, v1.4s, v2.4s       // encoding: [0x20,0xcc,0xa2,0x4e]
// CHECK: fmls v0.2d, v1.2d, v2.2d       // encoding: [0x20,0xcc,0xe2,0x4e]

