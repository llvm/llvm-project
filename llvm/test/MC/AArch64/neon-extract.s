// RUN: llvm-mc -triple=arm64 -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//------------------------------------------------------------------------------
// Instructions for bitwise extract
//------------------------------------------------------------------------------

        ext v0.8b, v1.8b, v2.8b, #0x3
        ext v0.16b, v1.16b, v2.16b, #0x3

// CHECK: ext	v0.8b, v1.8b, v2.8b, #{{0x3|3}}  // encoding: [0x20,0x18,0x02,0x2e]
// CHECK: ext	v0.16b, v1.16b, v2.16b, #{{0x3|3}} // encoding: [0x20,0x18,0x02,0x6e]
