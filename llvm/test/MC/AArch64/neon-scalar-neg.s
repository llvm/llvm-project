// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//----------------------------------------------------------------------
// Scalar Negate
//----------------------------------------------------------------------

    neg d29, d24

// CHECK: neg d29, d24    // encoding: [0x1d,0xbb,0xe0,0x7e]
        
//----------------------------------------------------------------------
// Scalar Signed Saturating Negate
//----------------------------------------------------------------------

    sqneg b19, b14
    sqneg h21, h15
    sqneg s20, s12
    sqneg d18, d12

// CHECK: sqneg b19, b14    // encoding: [0xd3,0x79,0x20,0x7e]
// CHECK: sqneg h21, h15    // encoding: [0xf5,0x79,0x60,0x7e]
// CHECK: sqneg s20, s12    // encoding: [0x94,0x79,0xa0,0x7e]
// CHECK: sqneg d18, d12    // encoding: [0x92,0x79,0xe0,0x7e]
