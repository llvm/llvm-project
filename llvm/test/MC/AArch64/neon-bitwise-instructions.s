// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//------------------------------------------------------------------------------
// Vector And
//------------------------------------------------------------------------------
         and v0.8b, v1.8b, v2.8b
         and v0.16b, v1.16b, v2.16b

// CHECK: and v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x1c,0x22,0x0e]
// CHECK: and v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x1c,0x22,0x4e]


//------------------------------------------------------------------------------
// Vector Orr
//------------------------------------------------------------------------------
         orr v0.8b, v1.8b, v2.8b
         orr v0.16b, v1.16b, v2.16b

// CHECK: orr v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x1c,0xa2,0x0e]
// CHECK: orr v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x1c,0xa2,0x4e]


//------------------------------------------------------------------------------
// Vector Eor
//------------------------------------------------------------------------------
         eor v0.8b, v1.8b, v2.8b
         eor v0.16b, v1.16b, v2.16b

// CHECK: eor v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x1c,0x22,0x2e]
// CHECK: eor v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x1c,0x22,0x6e]


//----------------------------------------------------------------------
// Vector Bitwise
//----------------------------------------------------------------------

         bit v0.8b, v1.8b, v2.8b
         bit v0.16b, v1.16b, v2.16b
         bif v0.8b, v1.8b, v2.8b
         bif v0.16b, v1.16b, v2.16b
         bsl v0.8b, v1.8b, v2.8b
         bsl v0.16b, v1.16b, v2.16b
         orn v0.8b, v1.8b, v2.8b
         orn v0.16b, v1.16b, v2.16b
         bic v0.8b, v1.8b, v2.8b
         bic v0.16b, v1.16b, v2.16b

// CHECK: bit v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x1c,0xa2,0x2e]
// CHECK: bit v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x1c,0xa2,0x6e]
// CHECK: bif v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x1c,0xe2,0x2e]
// CHECK: bif v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x1c,0xe2,0x6e]
// CHECK: bsl v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x1c,0x62,0x2e]
// CHECK: bsl v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x1c,0x62,0x6e]
// CHECK: orn v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x1c,0xe2,0x0e]
// CHECK: orn v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x1c,0xe2,0x4e]
// CHECK: bic v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x1c,0x62,0x0e]
// CHECK: bic v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x1c,0x62,0x4e]

