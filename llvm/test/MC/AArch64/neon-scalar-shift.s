// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

//------------------------------------------------------------------------------
// Scalar Integer Shift Lef (Signed)
//------------------------------------------------------------------------------
         sshl d17, d31, d8

// CHECK: sshl d17, d31, d8      // encoding: [0xf1,0x47,0xe8,0x5e]

//------------------------------------------------------------------------------
// Scalar Integer Shift Lef (Unsigned)
//------------------------------------------------------------------------------
         ushl d17, d31, d8

// CHECK: ushl d17, d31, d8      // encoding: [0xf1,0x47,0xe8,0x7e]

