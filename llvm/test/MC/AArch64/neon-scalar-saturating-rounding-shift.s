// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

//------------------------------------------------------------------------------
// Scalar Integer Saturating Rounding Shift Lef (Signed)
//------------------------------------------------------------------------------
         sqrshl b0, b1, b2
         sqrshl h10, h11, h12
         sqrshl s20, s21, s2
         sqrshl d17, d31, d8

// CHECK: sqrshl b0, b1, b2        // encoding: [0x20,0x5c,0x22,0x5e]
// CHECK: sqrshl h10, h11, h12     // encoding: [0x6a,0x5d,0x6c,0x5e]
// CHECK: sqrshl s20, s21, s2      // encoding: [0xb4,0x5e,0xa2,0x5e]
// CHECK: sqrshl d17, d31, d8      // encoding: [0xf1,0x5f,0xe8,0x5e]

//------------------------------------------------------------------------------
// Scalar Integer Saturating Rounding Shift Lef (Unsigned)
//------------------------------------------------------------------------------
         uqrshl b0, b1, b2
         uqrshl h10, h11, h12
         uqrshl s20, s21, s2
         uqrshl d17, d31, d8

// CHECK: uqrshl b0, b1, b2        // encoding: [0x20,0x5c,0x22,0x7e]
// CHECK: uqrshl h10, h11, h12     // encoding: [0x6a,0x5d,0x6c,0x7e]
// CHECK: uqrshl s20, s21, s2      // encoding: [0xb4,0x5e,0xa2,0x7e]
// CHECK: uqrshl d17, d31, d8      // encoding: [0xf1,0x5f,0xe8,0x7e]

