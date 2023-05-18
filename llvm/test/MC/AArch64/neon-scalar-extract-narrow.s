// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//----------------------------------------------------------------------
// Scalar Signed Saturating Extract Unsigned Narrow
//----------------------------------------------------------------------

    sqxtun b19, h14
    sqxtun h21, s15
    sqxtun s20, d12

// CHECK: sqxtun b19, h14  // encoding: [0xd3,0x29,0x21,0x7e]
// CHECK: sqxtun h21, s15  // encoding: [0xf5,0x29,0x61,0x7e]
// CHECK: sqxtun s20, d12  // encoding: [0x94,0x29,0xa1,0x7e]

//----------------------------------------------------------------------
// Scalar Signed Saturating Extract Signed Narrow
//----------------------------------------------------------------------

    sqxtn b18, h18
    sqxtn h20, s17
    sqxtn s19, d14

// CHECK: sqxtn b18, h18  // encoding: [0x52,0x4a,0x21,0x5e]
// CHECK: sqxtn h20, s17  // encoding: [0x34,0x4a,0x61,0x5e]
// CHECK: sqxtn s19, d14  // encoding: [0xd3,0x49,0xa1,0x5e]


//----------------------------------------------------------------------
// Scalar Unsigned Saturating Extract Narrow
//----------------------------------------------------------------------

    uqxtn b18, h18
    uqxtn h20, s17
    uqxtn s19, d14

// CHECK: uqxtn b18, h18  // encoding: [0x52,0x4a,0x21,0x7e]
// CHECK: uqxtn h20, s17  // encoding: [0x34,0x4a,0x61,0x7e]
// CHECK: uqxtn s19, d14  // encoding: [0xd3,0x49,0xa1,0x7e]
