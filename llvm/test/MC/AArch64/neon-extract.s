// RUN: llvm-mc -triple=arm64 -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//------------------------------------------------------------------------------
// Instructions for bitwise extract
//------------------------------------------------------------------------------

        ext v0.8b, v1.8b, v2.8b, #0x3
        ext v0.16b, v1.16b, v2.16b, #0x3

// CHECK: ext	v0.8b, v1.8b, v2.8b, #{{0x3|3}}  // encoding: [0x20,0x18,0x02,0x2e]
// CHECK: ext	v0.16b, v1.16b, v2.16b, #{{0x3|3}} // encoding: [0x20,0x18,0x02,0x6e]

// The index is four bits wide, but only three of them are available on the
// 64-bit form. Check both ends of each range.

        ext v0.8b, v1.8b, v2.8b, #0
        ext v0.8b, v1.8b, v2.8b, #7
        ext v0.16b, v1.16b, v2.16b, #0
        ext v0.16b, v1.16b, v2.16b, #15

// CHECK: ext	v0.8b, v1.8b, v2.8b, #{{0x0|0}}  // encoding: [0x20,0x00,0x02,0x2e]
// CHECK: ext	v0.8b, v1.8b, v2.8b, #{{0x7|7}}  // encoding: [0x20,0x38,0x02,0x2e]
// CHECK: ext	v0.16b, v1.16b, v2.16b, #{{0x0|0}} // encoding: [0x20,0x00,0x02,0x6e]
// CHECK: ext	v0.16b, v1.16b, v2.16b, #{{0xf|15}} // encoding: [0x20,0x78,0x02,0x6e]
