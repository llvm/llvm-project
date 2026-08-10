// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64


//------------------------------------------------------------------------------
// Vector Integer Halving Add (Signed)
//------------------------------------------------------------------------------
         shadd v0.8b, v1.8b, v2.8b
         shadd v0.16b, v1.16b, v2.16b
         shadd v0.4h, v1.4h, v2.4h
         shadd v0.8h, v1.8h, v2.8h
         shadd v0.2s, v1.2s, v2.2s
         shadd v0.4s, v1.4s, v2.4s

// CHECK: shadd v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x04,0x22,0x0e]
// CHECK: shadd v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x04,0x22,0x4e]
// CHECK: shadd v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x04,0x62,0x0e]
// CHECK: shadd v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x04,0x62,0x4e]
// CHECK: shadd v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x04,0xa2,0x0e]
// CHECK: shadd v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x04,0xa2,0x4e]


//------------------------------------------------------------------------------
// Vector Integer Halving Add (Unsigned)
//------------------------------------------------------------------------------
         uhadd v0.8b, v1.8b, v2.8b
         uhadd v0.16b, v1.16b, v2.16b
         uhadd v0.4h, v1.4h, v2.4h
         uhadd v0.8h, v1.8h, v2.8h
         uhadd v0.2s, v1.2s, v2.2s
         uhadd v0.4s, v1.4s, v2.4s

// CHECK: uhadd v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x04,0x22,0x2e]
// CHECK: uhadd v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x04,0x22,0x6e]
// CHECK: uhadd v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x04,0x62,0x2e]
// CHECK: uhadd v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x04,0x62,0x6e]
// CHECK: uhadd v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x04,0xa2,0x2e]
// CHECK: uhadd v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x04,0xa2,0x6e]

//------------------------------------------------------------------------------
// Vector Integer Halving Sub (Signed)
//------------------------------------------------------------------------------
         shsub v0.8b, v1.8b, v2.8b
         shsub v0.16b, v1.16b, v2.16b
         shsub v0.4h, v1.4h, v2.4h
         shsub v0.8h, v1.8h, v2.8h
         shsub v0.2s, v1.2s, v2.2s
         shsub v0.4s, v1.4s, v2.4s

// CHECK: shsub v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x24,0x22,0x0e]
// CHECK: shsub v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x24,0x22,0x4e]
// CHECK: shsub v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x24,0x62,0x0e]
// CHECK: shsub v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x24,0x62,0x4e]
// CHECK: shsub v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x24,0xa2,0x0e]
// CHECK: shsub v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x24,0xa2,0x4e]

//------------------------------------------------------------------------------
// Vector Integer Halving Sub (Unsigned)
//------------------------------------------------------------------------------
         uhsub v0.8b, v1.8b, v2.8b
         uhsub v0.16b, v1.16b, v2.16b
         uhsub v0.4h, v1.4h, v2.4h
         uhsub v0.8h, v1.8h, v2.8h
         uhsub v0.2s, v1.2s, v2.2s
         uhsub v0.4s, v1.4s, v2.4s

// CHECK: uhsub v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x24,0x22,0x2e]
// CHECK: uhsub v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x24,0x22,0x6e]
// CHECK: uhsub v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x24,0x62,0x2e]
// CHECK: uhsub v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x24,0x62,0x6e]
// CHECK: uhsub v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x24,0xa2,0x2e]
// CHECK: uhsub v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x24,0xa2,0x6e]

