// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon,+fullfp16 -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//----------------------------------------------------------------------
// Scalar Absolute Value
//----------------------------------------------------------------------

    abs d29, d24

// CHECK: abs d29, d24    // encoding: [0x1d,0xbb,0xe0,0x5e]
        
//----------------------------------------------------------------------
// Scalar Floating-point Absolute Difference
//----------------------------------------------------------------------

    fabd h29, h24, h20
    fabd s29, s24, s20
    fabd d29, d24, d20

// CHECK: fabd    h29, h24, h20           // encoding: [0x1d,0x17,0xd4,0x7e]
// CHECK: fabd s29, s24, s20  // encoding: [0x1d,0xd7,0xb4,0x7e]
// CHECK: fabd d29, d24, d20  // encoding: [0x1d,0xd7,0xf4,0x7e]

//----------------------------------------------------------------------
// Scalar Signed Saturating Absolute Value
//----------------------------------------------------------------------

    sqabs b19, b14
    sqabs h21, h15
    sqabs s20, s12
    sqabs d18, d12

// CHECK: sqabs b19, b14    // encoding: [0xd3,0x79,0x20,0x5e]
// CHECK: sqabs h21, h15    // encoding: [0xf5,0x79,0x60,0x5e]
// CHECK: sqabs s20, s12    // encoding: [0x94,0x79,0xa0,0x5e]
// CHECK: sqabs d18, d12    // encoding: [0x92,0x79,0xe0,0x5e]
