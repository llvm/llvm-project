// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon,+fullfp16 -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64


//------------------------------------------------------------------------------
// Vector  Add Pairwise (Integer)
//------------------------------------------------------------------------------
         addp v0.8b, v1.8b, v2.8b
         addp v0.16b, v1.16b, v2.16b
         addp v0.4h, v1.4h, v2.4h
         addp v0.8h, v1.8h, v2.8h
         addp v0.2s, v1.2s, v2.2s
         addp v0.4s, v1.4s, v2.4s
         addp v0.2d, v1.2d, v2.2d

// CHECK: addp v0.8b, v1.8b, v2.8b        // encoding: [0x20,0xbc,0x22,0x0e]
// CHECK: addp v0.16b, v1.16b, v2.16b     // encoding: [0x20,0xbc,0x22,0x4e]
// CHECK: addp v0.4h, v1.4h, v2.4h        // encoding: [0x20,0xbc,0x62,0x0e]
// CHECK: addp v0.8h, v1.8h, v2.8h        // encoding: [0x20,0xbc,0x62,0x4e]
// CHECK: addp v0.2s, v1.2s, v2.2s        // encoding: [0x20,0xbc,0xa2,0x0e]
// CHECK: addp v0.4s, v1.4s, v2.4s        // encoding: [0x20,0xbc,0xa2,0x4e]
// CHECK: addp v0.2d, v1.2d, v2.2d        // encoding: [0x20,0xbc,0xe2,0x4e]

//------------------------------------------------------------------------------
// Vector Add Pairwise (Floating Point
//------------------------------------------------------------------------------
         faddp v0.4h, v1.4h, v2.4h
         faddp v0.8h, v1.8h, v2.8h
         faddp v0.2s, v1.2s, v2.2s
         faddp v0.4s, v1.4s, v2.4s
         faddp v0.2d, v1.2d, v2.2d

// CHECK: faddp   v0.4h, v1.4h, v2.4h     // encoding: [0x20,0x14,0x42,0x2e]
// CHECK: faddp   v0.8h, v1.8h, v2.8h     // encoding: [0x20,0x14,0x42,0x6e]
// CHECK: faddp v0.2s, v1.2s, v2.2s       // encoding: [0x20,0xd4,0x22,0x2e]
// CHECK: faddp v0.4s, v1.4s, v2.4s       // encoding: [0x20,0xd4,0x22,0x6e]
// CHECK: faddp v0.2d, v1.2d, v2.2d       // encoding: [0x20,0xd4,0x62,0x6e]
