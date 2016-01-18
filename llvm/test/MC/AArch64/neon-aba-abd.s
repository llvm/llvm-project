// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//----------------------------------------------------------------------
// Vector Absolute Difference and Accumulate (Signed, Unsigned)
//----------------------------------------------------------------------
         uaba v0.8b, v1.8b, v2.8b
         uaba v0.16b, v1.16b, v2.16b
         uaba v0.4h, v1.4h, v2.4h
         uaba v0.8h, v1.8h, v2.8h
         uaba v0.2s, v1.2s, v2.2s
         uaba v0.4s, v1.4s, v2.4s

// CHECK: uaba v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x7c,0x22,0x2e]
// CHECK: uaba v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x7c,0x22,0x6e]
// CHECK: uaba v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x7c,0x62,0x2e]
// CHECK: uaba v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x7c,0x62,0x6e]
// CHECK: uaba v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x7c,0xa2,0x2e]
// CHECK: uaba v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x7c,0xa2,0x6e]


         saba v0.8b, v1.8b, v2.8b
         saba v0.16b, v1.16b, v2.16b
         saba v0.4h, v1.4h, v2.4h
         saba v0.8h, v1.8h, v2.8h
         saba v0.2s, v1.2s, v2.2s
         saba v0.4s, v1.4s, v2.4s

// CHECK: saba v0.8b, v1.8b, v2.8b         // encoding: [0x20,0x7c,0x22,0x0e]
// CHECK: saba v0.16b, v1.16b, v2.16b      // encoding: [0x20,0x7c,0x22,0x4e]
// CHECK: saba v0.4h, v1.4h, v2.4h         // encoding: [0x20,0x7c,0x62,0x0e]
// CHECK: saba v0.8h, v1.8h, v2.8h         // encoding: [0x20,0x7c,0x62,0x4e]
// CHECK: saba v0.2s, v1.2s, v2.2s         // encoding: [0x20,0x7c,0xa2,0x0e]
// CHECK: saba v0.4s, v1.4s, v2.4s         // encoding: [0x20,0x7c,0xa2,0x4e]

//----------------------------------------------------------------------
// Vector Absolute Difference (Signed, Unsigned)
//----------------------------------------------------------------------
         uabd v0.8b, v1.8b, v2.8b
         uabd v0.16b, v1.16b, v2.16b
         uabd v0.4h, v1.4h, v2.4h
         uabd v0.8h, v1.8h, v2.8h
         uabd v0.2s, v1.2s, v2.2s
         uabd v0.4s, v1.4s, v2.4s

// CHECK: uabd v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x74,0x22,0x2e]
// CHECK: uabd v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x74,0x22,0x6e]
// CHECK: uabd v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x74,0x62,0x2e]
// CHECK: uabd v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x74,0x62,0x6e]
// CHECK: uabd v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x74,0xa2,0x2e]
// CHECK: uabd v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x74,0xa2,0x6e]

         sabd v0.8b, v1.8b, v2.8b
         sabd v0.16b, v1.16b, v2.16b
         sabd v0.4h, v1.4h, v2.4h
         sabd v0.8h, v1.8h, v2.8h
         sabd v0.2s, v1.2s, v2.2s
         sabd v0.4s, v1.4s, v2.4s

// CHECK: sabd v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x74,0x22,0x0e]
// CHECK: sabd v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x74,0x22,0x4e]
// CHECK: sabd v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x74,0x62,0x0e]
// CHECK: sabd v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x74,0x62,0x4e]
// CHECK: sabd v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x74,0xa2,0x0e]
// CHECK: sabd v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x74,0xa2,0x4e]

//----------------------------------------------------------------------
// Vector Absolute Difference (Floating Point)
//----------------------------------------------------------------------
         fabd v0.2s, v1.2s, v2.2s
         fabd v31.4s, v15.4s, v16.4s
         fabd v7.2d, v8.2d, v25.2d

// CHECK: fabd v0.2s, v1.2s, v2.2s    // encoding: [0x20,0xd4,0xa2,0x2e]
// CHECK: fabd v31.4s, v15.4s, v16.4s // encoding: [0xff,0xd5,0xb0,0x6e]
// CHECK: fabd v7.2d, v8.2d, v25.2d   // encoding: [0x07,0xd5,0xf9,0x6e]

