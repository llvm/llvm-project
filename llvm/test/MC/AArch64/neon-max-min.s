// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon,+fullfp16 -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//----------------------------------------------------------------------
// Vector Maximum (Signed and Unsigned Integer)
//----------------------------------------------------------------------
         smax v0.8b, v1.8b, v2.8b
         smax v0.16b, v1.16b, v2.16b
         smax v0.4h, v1.4h, v2.4h
         smax v0.8h, v1.8h, v2.8h
         smax v0.2s, v1.2s, v2.2s
         smax v0.4s, v1.4s, v2.4s

// CHECK: smax v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x64,0x22,0x0e]
// CHECK: smax v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x64,0x22,0x4e]
// CHECK: smax v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x64,0x62,0x0e]
// CHECK: smax v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x64,0x62,0x4e]
// CHECK: smax v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x64,0xa2,0x0e]
// CHECK: smax v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x64,0xa2,0x4e]

         umax v0.8b, v1.8b, v2.8b
         umax v0.16b, v1.16b, v2.16b
         umax v0.4h, v1.4h, v2.4h
         umax v0.8h, v1.8h, v2.8h
         umax v0.2s, v1.2s, v2.2s
         umax v0.4s, v1.4s, v2.4s

// CHECK: umax v0.8b, v1.8b, v2.8b         // encoding: [0x20,0x64,0x22,0x2e]
// CHECK: umax v0.16b, v1.16b, v2.16b      // encoding: [0x20,0x64,0x22,0x6e]
// CHECK: umax v0.4h, v1.4h, v2.4h         // encoding: [0x20,0x64,0x62,0x2e]
// CHECK: umax v0.8h, v1.8h, v2.8h         // encoding: [0x20,0x64,0x62,0x6e]
// CHECK: umax v0.2s, v1.2s, v2.2s         // encoding: [0x20,0x64,0xa2,0x2e]
// CHECK: umax v0.4s, v1.4s, v2.4s         // encoding: [0x20,0x64,0xa2,0x6e]

//----------------------------------------------------------------------
// Vector Minimum (Signed and Unsigned Integer)
//----------------------------------------------------------------------
         smin v0.8b, v1.8b, v2.8b
         smin v0.16b, v1.16b, v2.16b
         smin v0.4h, v1.4h, v2.4h
         smin v0.8h, v1.8h, v2.8h
         smin v0.2s, v1.2s, v2.2s
         smin v0.4s, v1.4s, v2.4s

// CHECK: smin v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x6c,0x22,0x0e]
// CHECK: smin v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x6c,0x22,0x4e]
// CHECK: smin v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x6c,0x62,0x0e]
// CHECK: smin v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x6c,0x62,0x4e]
// CHECK: smin v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x6c,0xa2,0x0e]
// CHECK: smin v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x6c,0xa2,0x4e]

         umin v0.8b, v1.8b, v2.8b
         umin v0.16b, v1.16b, v2.16b
         umin v0.4h, v1.4h, v2.4h
         umin v0.8h, v1.8h, v2.8h
         umin v0.2s, v1.2s, v2.2s
         umin v0.4s, v1.4s, v2.4s

// CHECK: umin v0.8b, v1.8b, v2.8b         // encoding: [0x20,0x6c,0x22,0x2e]
// CHECK: umin v0.16b, v1.16b, v2.16b      // encoding: [0x20,0x6c,0x22,0x6e]
// CHECK: umin v0.4h, v1.4h, v2.4h         // encoding: [0x20,0x6c,0x62,0x2e]
// CHECK: umin v0.8h, v1.8h, v2.8h         // encoding: [0x20,0x6c,0x62,0x6e]
// CHECK: umin v0.2s, v1.2s, v2.2s         // encoding: [0x20,0x6c,0xa2,0x2e]
// CHECK: umin v0.4s, v1.4s, v2.4s         // encoding: [0x20,0x6c,0xa2,0x6e]

//----------------------------------------------------------------------
// Vector Maximum (Floating Point)
//----------------------------------------------------------------------
         fmax v0.4h, v1.4h, v2.4h
         fmax v0.8h, v1.8h, v2.8h
         fmax v0.2s, v1.2s, v2.2s
         fmax v31.4s, v15.4s, v16.4s
         fmax v7.2d, v8.2d, v25.2d

// CHECK: fmax    v0.4h, v1.4h, v2.4h     // encoding: [0x20,0x34,0x42,0x0e]
// CHECK: fmax    v0.8h, v1.8h, v2.8h     // encoding: [0x20,0x34,0x42,0x4e]
// CHECK: fmax v0.2s, v1.2s, v2.2s    // encoding: [0x20,0xf4,0x22,0x0e]
// CHECK: fmax v31.4s, v15.4s, v16.4s // encoding: [0xff,0xf5,0x30,0x4e]
// CHECK: fmax v7.2d, v8.2d, v25.2d   // encoding: [0x07,0xf5,0x79,0x4e]

//----------------------------------------------------------------------
// Vector Minimum (Floating Point)
//----------------------------------------------------------------------
         fmin v10.4h, v15.4h, v22.4h
         fmin v10.8h, v15.8h, v22.8h
         fmin v10.2s, v15.2s, v22.2s
         fmin v3.4s, v5.4s, v6.4s
         fmin v17.2d, v13.2d, v2.2d

// CHECK: fmin    v10.4h, v15.4h, v22.4h  // encoding: [0xea,0x35,0xd6,0x0e]
// CHECK: fmin    v10.8h, v15.8h, v22.8h  // encoding: [0xea,0x35,0xd6,0x4e]
// CHECK: fmin v10.2s, v15.2s, v22.2s  // encoding: [0xea,0xf5,0xb6,0x0e]
// CHECK: fmin v3.4s, v5.4s, v6.4s     // encoding: [0xa3,0xf4,0xa6,0x4e]
// CHECK: fmin v17.2d, v13.2d, v2.2d   // encoding: [0xb1,0xf5,0xe2,0x4e]

//----------------------------------------------------------------------
// Vector maxNum (Floating Point)
//----------------------------------------------------------------------
         fmaxnm v0.4h, v1.4h, v2.4h
         fmaxnm v0.8h, v1.8h, v2.8h
         fmaxnm v0.2s, v1.2s, v2.2s
         fmaxnm v31.4s, v15.4s, v16.4s
         fmaxnm v7.2d, v8.2d, v25.2d

// CHECK: fmaxnm  v0.4h, v1.4h, v2.4h     // encoding: [0x20,0x04,0x42,0x0e]
// CHECK: fmaxnm  v0.8h, v1.8h, v2.8h     // encoding: [0x20,0x04,0x42,0x4e]
// CHECK: fmaxnm v0.2s, v1.2s, v2.2s    // encoding: [0x20,0xc4,0x22,0x0e]
// CHECK: fmaxnm v31.4s, v15.4s, v16.4s // encoding: [0xff,0xc5,0x30,0x4e]
// CHECK: fmaxnm v7.2d, v8.2d, v25.2d   // encoding: [0x07,0xc5,0x79,0x4e]

//----------------------------------------------------------------------
// Vector minNum (Floating Point)
//----------------------------------------------------------------------
         fminnm v10.4h, v15.4h, v22.4h
         fminnm v10.8h, v15.8h, v22.8h
         fminnm v10.2s, v15.2s, v22.2s
         fminnm v3.4s, v5.4s, v6.4s
         fminnm v17.2d, v13.2d, v2.2d

// CHECK: fminnm  v10.4h, v15.4h, v22.4h  // encoding: [0xea,0x05,0xd6,0x0e]
// CHECK: fminnm  v10.8h, v15.8h, v22.8h  // encoding: [0xea,0x05,0xd6,0x4e]
// CHECK: fminnm v10.2s, v15.2s, v22.2s  // encoding: [0xea,0xc5,0xb6,0x0e]
// CHECK: fminnm v3.4s, v5.4s, v6.4s     // encoding: [0xa3,0xc4,0xa6,0x4e]
// CHECK: fminnm v17.2d, v13.2d, v2.2d   // encoding: [0xb1,0xc5,0xe2,0x4e]

