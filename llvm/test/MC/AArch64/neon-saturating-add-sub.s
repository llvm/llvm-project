// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64


//------------------------------------------------------------------------------
// Vector Integer Saturating Add (Signed)
//------------------------------------------------------------------------------
         sqadd v0.8b, v1.8b, v2.8b
         sqadd v0.16b, v1.16b, v2.16b
         sqadd v0.4h, v1.4h, v2.4h
         sqadd v0.8h, v1.8h, v2.8h
         sqadd v0.2s, v1.2s, v2.2s
         sqadd v0.4s, v1.4s, v2.4s
         sqadd v0.2d, v1.2d, v2.2d

// CHECK: sqadd v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x0c,0x22,0x0e]
// CHECK: sqadd v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x0c,0x22,0x4e]
// CHECK: sqadd v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x0c,0x62,0x0e]
// CHECK: sqadd v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x0c,0x62,0x4e]
// CHECK: sqadd v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x0c,0xa2,0x0e]
// CHECK: sqadd v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x0c,0xa2,0x4e]
// CHECK: sqadd v0.2d, v1.2d, v2.2d        // encoding: [0x20,0x0c,0xe2,0x4e]

//------------------------------------------------------------------------------
// Vector Integer Saturating Add (Unsigned)
//------------------------------------------------------------------------------
         uqadd v0.8b, v1.8b, v2.8b
         uqadd v0.16b, v1.16b, v2.16b
         uqadd v0.4h, v1.4h, v2.4h
         uqadd v0.8h, v1.8h, v2.8h
         uqadd v0.2s, v1.2s, v2.2s
         uqadd v0.4s, v1.4s, v2.4s
         uqadd v0.2d, v1.2d, v2.2d

// CHECK: uqadd v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x0c,0x22,0x2e]
// CHECK: uqadd v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x0c,0x22,0x6e]
// CHECK: uqadd v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x0c,0x62,0x2e]
// CHECK: uqadd v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x0c,0x62,0x6e]
// CHECK: uqadd v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x0c,0xa2,0x2e]
// CHECK: uqadd v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x0c,0xa2,0x6e]
// CHECK: uqadd v0.2d, v1.2d, v2.2d        // encoding: [0x20,0x0c,0xe2,0x6e]

//------------------------------------------------------------------------------
// Vector Integer Saturating Sub (Signed)
//------------------------------------------------------------------------------
         sqsub v0.8b, v1.8b, v2.8b
         sqsub v0.16b, v1.16b, v2.16b
         sqsub v0.4h, v1.4h, v2.4h
         sqsub v0.8h, v1.8h, v2.8h
         sqsub v0.2s, v1.2s, v2.2s
         sqsub v0.4s, v1.4s, v2.4s
         sqsub v0.2d, v1.2d, v2.2d

// CHECK: sqsub v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x2c,0x22,0x0e]
// CHECK: sqsub v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x2c,0x22,0x4e]
// CHECK: sqsub v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x2c,0x62,0x0e]
// CHECK: sqsub v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x2c,0x62,0x4e]
// CHECK: sqsub v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x2c,0xa2,0x0e]
// CHECK: sqsub v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x2c,0xa2,0x4e]
// CHECK: sqsub v0.2d, v1.2d, v2.2d        // encoding: [0x20,0x2c,0xe2,0x4e]

//------------------------------------------------------------------------------
// Vector Integer Saturating Sub (Unsigned)
//------------------------------------------------------------------------------
         uqsub v0.8b, v1.8b, v2.8b
         uqsub v0.16b, v1.16b, v2.16b
         uqsub v0.4h, v1.4h, v2.4h
         uqsub v0.8h, v1.8h, v2.8h
         uqsub v0.2s, v1.2s, v2.2s
         uqsub v0.4s, v1.4s, v2.4s
         uqsub v0.2d, v1.2d, v2.2d

// CHECK: uqsub v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x2c,0x22,0x2e]
// CHECK: uqsub v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x2c,0x22,0x6e]
// CHECK: uqsub v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x2c,0x62,0x2e]
// CHECK: uqsub v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x2c,0x62,0x6e]
// CHECK: uqsub v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x2c,0xa2,0x2e]
// CHECK: uqsub v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x2c,0xa2,0x6e]
// CHECK: uqsub v0.2d, v1.2d, v2.2d        // encoding: [0x20,0x2c,0xe2,0x6e]


