// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64


//------------------------------------------------------------------------------
// Vector Integer Rouding Halving Add (Signed)
//------------------------------------------------------------------------------
         srhadd v0.8b, v1.8b, v2.8b
         srhadd v0.16b, v1.16b, v2.16b
         srhadd v0.4h, v1.4h, v2.4h
         srhadd v0.8h, v1.8h, v2.8h
         srhadd v0.2s, v1.2s, v2.2s
         srhadd v0.4s, v1.4s, v2.4s

// CHECK: srhadd v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x14,0x22,0x0e]
// CHECK: srhadd v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x14,0x22,0x4e]
// CHECK: srhadd v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x14,0x62,0x0e]
// CHECK: srhadd v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x14,0x62,0x4e]
// CHECK: srhadd v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x14,0xa2,0x0e]
// CHECK: srhadd v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x14,0xa2,0x4e]

//------------------------------------------------------------------------------
// Vector Integer Rouding Halving Add (Unsigned)
//------------------------------------------------------------------------------
         urhadd v0.8b, v1.8b, v2.8b
         urhadd v0.16b, v1.16b, v2.16b
         urhadd v0.4h, v1.4h, v2.4h
         urhadd v0.8h, v1.8h, v2.8h
         urhadd v0.2s, v1.2s, v2.2s
         urhadd v0.4s, v1.4s, v2.4s

// CHECK: urhadd v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x14,0x22,0x2e]
// CHECK: urhadd v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x14,0x22,0x6e]
// CHECK: urhadd v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x14,0x62,0x2e]
// CHECK: urhadd v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x14,0x62,0x6e]
// CHECK: urhadd v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x14,0xa2,0x2e]
// CHECK: urhadd v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x14,0xa2,0x6e]

