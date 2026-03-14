// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s


//------------------------------------------------------------------------------
// Scalar Integer Rounding Shift Lef (Signed)
//------------------------------------------------------------------------------
         srshl d17, d31, d8

// CHECK: srshl d17, d31, d8      // encoding: [0xf1,0x57,0xe8,0x5e]

//------------------------------------------------------------------------------
// Scalar Integer Rounding Shift Lef (Unsigned)
//------------------------------------------------------------------------------
         urshl d17, d31, d8

// CHECK: urshl d17, d31, d8      // encoding: [0xf1,0x57,0xe8,0x7e]

