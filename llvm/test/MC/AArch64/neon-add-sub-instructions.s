// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64


//------------------------------------------------------------------------------
// Vector Integer Add
//------------------------------------------------------------------------------
         add v0.8b, v1.8b, v2.8b
         add v0.16b, v1.16b, v2.16b
         add v0.4h, v1.4h, v2.4h
         add v0.8h, v1.8h, v2.8h
         add v0.2s, v1.2s, v2.2s
         add v0.4s, v1.4s, v2.4s
         add v0.2d, v1.2d, v2.2d

// CHECK: add v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x84,0x22,0x0e]
// CHECK: add v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x84,0x22,0x4e]
// CHECK: add v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x84,0x62,0x0e]
// CHECK: add v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x84,0x62,0x4e]
// CHECK: add v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x84,0xa2,0x0e]
// CHECK: add v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x84,0xa2,0x4e]
// CHECK: add v0.2d, v1.2d, v2.2d        // encoding: [0x20,0x84,0xe2,0x4e]

//------------------------------------------------------------------------------
// Vector Integer Sub
//------------------------------------------------------------------------------
         sub v0.8b, v1.8b, v2.8b
         sub v0.16b, v1.16b, v2.16b
         sub v0.4h, v1.4h, v2.4h
         sub v0.8h, v1.8h, v2.8h
         sub v0.2s, v1.2s, v2.2s
         sub v0.4s, v1.4s, v2.4s
         sub v0.2d, v1.2d, v2.2d

// CHECK: sub v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x84,0x22,0x2e]
// CHECK: sub v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x84,0x22,0x6e]
// CHECK: sub v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x84,0x62,0x2e]
// CHECK: sub v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x84,0x62,0x6e]
// CHECK: sub v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x84,0xa2,0x2e]
// CHECK: sub v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x84,0xa2,0x6e]
// CHECK: sub v0.2d, v1.2d, v2.2d        // encoding: [0x20,0x84,0xe2,0x6e]

//------------------------------------------------------------------------------
// Vector Floating-Point Add
//------------------------------------------------------------------------------
         fadd v0.2s, v1.2s, v2.2s
         fadd v0.4s, v1.4s, v2.4s
         fadd v0.2d, v1.2d, v2.2d

// CHECK: fadd v0.2s, v1.2s, v2.2s       // encoding: [0x20,0xd4,0x22,0x0e]
// CHECK: fadd v0.4s, v1.4s, v2.4s       // encoding: [0x20,0xd4,0x22,0x4e]
// CHECK: fadd v0.2d, v1.2d, v2.2d       // encoding: [0x20,0xd4,0x62,0x4e]


//------------------------------------------------------------------------------
// Vector Floating-Point Sub
//------------------------------------------------------------------------------
         fsub v0.2s, v1.2s, v2.2s
         fsub v0.4s, v1.4s, v2.4s
         fsub v0.2d, v1.2d, v2.2d

// CHECK: fsub v0.2s, v1.2s, v2.2s       // encoding: [0x20,0xd4,0xa2,0x0e]
// CHECK: fsub v0.4s, v1.4s, v2.4s       // encoding: [0x20,0xd4,0xa2,0x4e]
// CHECK: fsub v0.2d, v1.2d, v2.2d       // encoding: [0x20,0xd4,0xe2,0x4e]



