// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

//------------------------------------------------------------------------------
// Scalar Integer Saturating Shift Lef (Signed)
//------------------------------------------------------------------------------
         sqshl b0, b1, b2
         sqshl h10, h11, h12
         sqshl s20, s21, s2
         sqshl d17, d31, d8

// CHECK: sqshl b0, b1, b2        // encoding: [0x20,0x4c,0x22,0x5e]
// CHECK: sqshl h10, h11, h12     // encoding: [0x6a,0x4d,0x6c,0x5e]
// CHECK: sqshl s20, s21, s2      // encoding: [0xb4,0x4e,0xa2,0x5e]
// CHECK: sqshl d17, d31, d8      // encoding: [0xf1,0x4f,0xe8,0x5e]

//------------------------------------------------------------------------------
// Scalar Integer Saturating Shift Lef (Unsigned)
//------------------------------------------------------------------------------
         uqshl b0, b1, b2
         uqshl h10, h11, h12
         uqshl s20, s21, s2
         uqshl d17, d31, d8

// CHECK: uqshl b0, b1, b2        // encoding: [0x20,0x4c,0x22,0x7e]
// CHECK: uqshl h10, h11, h12     // encoding: [0x6a,0x4d,0x6c,0x7e]
// CHECK: uqshl s20, s21, s2      // encoding: [0xb4,0x4e,0xa2,0x7e]
// CHECK: uqshl d17, d31, d8      // encoding: [0xf1,0x4f,0xe8,0x7e]


