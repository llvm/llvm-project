// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon,+fullfp16 -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//----------------------------------------------------------------------
// Vector Maximum Pairwise (Signed and Unsigned Integer)
//----------------------------------------------------------------------
         smaxp v0.8b, v1.8b, v2.8b
         smaxp v0.16b, v1.16b, v2.16b
         smaxp v0.4h, v1.4h, v2.4h
         smaxp v0.8h, v1.8h, v2.8h
         smaxp v0.2s, v1.2s, v2.2s
         smaxp v0.4s, v1.4s, v2.4s

// CHECK: smaxp v0.8b, v1.8b, v2.8b        // encoding: [0x20,0xa4,0x22,0x0e]
// CHECK: smaxp v0.16b, v1.16b, v2.16b     // encoding: [0x20,0xa4,0x22,0x4e]
// CHECK: smaxp v0.4h, v1.4h, v2.4h        // encoding: [0x20,0xa4,0x62,0x0e]
// CHECK: smaxp v0.8h, v1.8h, v2.8h        // encoding: [0x20,0xa4,0x62,0x4e]
// CHECK: smaxp v0.2s, v1.2s, v2.2s        // encoding: [0x20,0xa4,0xa2,0x0e]
// CHECK: smaxp v0.4s, v1.4s, v2.4s        // encoding: [0x20,0xa4,0xa2,0x4e]

         umaxp v0.8b, v1.8b, v2.8b
         umaxp v0.16b, v1.16b, v2.16b
         umaxp v0.4h, v1.4h, v2.4h
         umaxp v0.8h, v1.8h, v2.8h
         umaxp v0.2s, v1.2s, v2.2s
         umaxp v0.4s, v1.4s, v2.4s

// CHECK: umaxp v0.8b, v1.8b, v2.8b         // encoding: [0x20,0xa4,0x22,0x2e]
// CHECK: umaxp v0.16b, v1.16b, v2.16b      // encoding: [0x20,0xa4,0x22,0x6e]
// CHECK: umaxp v0.4h, v1.4h, v2.4h         // encoding: [0x20,0xa4,0x62,0x2e]
// CHECK: umaxp v0.8h, v1.8h, v2.8h         // encoding: [0x20,0xa4,0x62,0x6e]
// CHECK: umaxp v0.2s, v1.2s, v2.2s         // encoding: [0x20,0xa4,0xa2,0x2e]
// CHECK: umaxp v0.4s, v1.4s, v2.4s         // encoding: [0x20,0xa4,0xa2,0x6e]

//----------------------------------------------------------------------
// Vector Minimum Pairwise (Signed and Unsigned Integer)
//----------------------------------------------------------------------
         sminp v0.8b, v1.8b, v2.8b
         sminp v0.16b, v1.16b, v2.16b
         sminp v0.4h, v1.4h, v2.4h
         sminp v0.8h, v1.8h, v2.8h
         sminp v0.2s, v1.2s, v2.2s
         sminp v0.4s, v1.4s, v2.4s

// CHECK: sminp v0.8b, v1.8b, v2.8b        // encoding: [0x20,0xac,0x22,0x0e]
// CHECK: sminp v0.16b, v1.16b, v2.16b     // encoding: [0x20,0xac,0x22,0x4e]
// CHECK: sminp v0.4h, v1.4h, v2.4h        // encoding: [0x20,0xac,0x62,0x0e]
// CHECK: sminp v0.8h, v1.8h, v2.8h        // encoding: [0x20,0xac,0x62,0x4e]
// CHECK: sminp v0.2s, v1.2s, v2.2s        // encoding: [0x20,0xac,0xa2,0x0e]
// CHECK: sminp v0.4s, v1.4s, v2.4s        // encoding: [0x20,0xac,0xa2,0x4e]

         uminp v0.8b, v1.8b, v2.8b
         uminp v0.16b, v1.16b, v2.16b
         uminp v0.4h, v1.4h, v2.4h
         uminp v0.8h, v1.8h, v2.8h
         uminp v0.2s, v1.2s, v2.2s
         uminp v0.4s, v1.4s, v2.4s

// CHECK: uminp v0.8b, v1.8b, v2.8b         // encoding: [0x20,0xac,0x22,0x2e]
// CHECK: uminp v0.16b, v1.16b, v2.16b      // encoding: [0x20,0xac,0x22,0x6e]
// CHECK: uminp v0.4h, v1.4h, v2.4h         // encoding: [0x20,0xac,0x62,0x2e]
// CHECK: uminp v0.8h, v1.8h, v2.8h         // encoding: [0x20,0xac,0x62,0x6e]
// CHECK: uminp v0.2s, v1.2s, v2.2s         // encoding: [0x20,0xac,0xa2,0x2e]
// CHECK: uminp v0.4s, v1.4s, v2.4s         // encoding: [0x20,0xac,0xa2,0x6e]

//----------------------------------------------------------------------
// Vector Maximum Pairwise (Floating Point)
//----------------------------------------------------------------------
         fmaxp v0.4h, v1.4h, v2.4h
         fmaxp v31.8h, v15.8h, v16.8h
         fmaxp v0.2s, v1.2s, v2.2s
         fmaxp v31.4s, v15.4s, v16.4s
         fmaxp v7.2d, v8.2d, v25.2d

// CHECK: fmaxp   v0.4h, v1.4h, v2.4h     // encoding: [0x20,0x34,0x42,0x2e]
// CHECK: fmaxp   v31.8h, v15.8h, v16.8h  // encoding: [0xff,0x35,0x50,0x6e]
// CHECK: fmaxp v0.2s, v1.2s, v2.2s    // encoding: [0x20,0xf4,0x22,0x2e]
// CHECK: fmaxp v31.4s, v15.4s, v16.4s // encoding: [0xff,0xf5,0x30,0x6e]
// CHECK: fmaxp v7.2d, v8.2d, v25.2d   // encoding: [0x07,0xf5,0x79,0x6e]

//----------------------------------------------------------------------
// Vector Minimum Pairwise (Floating Point)
//----------------------------------------------------------------------
         fminp v10.4h, v15.4h, v22.4h
         fminp v3.8h, v5.8h, v6.8h
         fminp v10.2s, v15.2s, v22.2s
         fminp v3.4s, v5.4s, v6.4s
         fminp v17.2d, v13.2d, v2.2d

// CHECK: fminp   v10.4h, v15.4h, v22.4h  // encoding: [0xea,0x35,0xd6,0x2e]
// CHECK: fminp   v3.8h, v5.8h, v6.8h     // encoding: [0xa3,0x34,0xc6,0x6e]
// CHECK: fminp v10.2s, v15.2s, v22.2s  // encoding: [0xea,0xf5,0xb6,0x2e]
// CHECK: fminp v3.4s, v5.4s, v6.4s     // encoding: [0xa3,0xf4,0xa6,0x6e]
// CHECK: fminp v17.2d, v13.2d, v2.2d   // encoding: [0xb1,0xf5,0xe2,0x6e]

//----------------------------------------------------------------------
// Vector maxNum Pairwise (Floating Point)
//----------------------------------------------------------------------
         fmaxnmp v0.4h, v1.4h, v2.4h
         fmaxnmp v31.8h, v15.8h, v16.8h
         fmaxnmp v0.2s, v1.2s, v2.2s
         fmaxnmp v31.4s, v15.4s, v16.4s
         fmaxnmp v7.2d, v8.2d, v25.2d

// CHECK: fmaxnmp v0.4h, v1.4h, v2.4h     // encoding: [0x20,0x04,0x42,0x2e]
// CHECK: fmaxnmp v31.8h, v15.8h, v16.8h  // encoding: [0xff,0x05,0x50,0x6e]
// CHECK: fmaxnmp v0.2s, v1.2s, v2.2s    // encoding: [0x20,0xc4,0x22,0x2e]
// CHECK: fmaxnmp v31.4s, v15.4s, v16.4s // encoding: [0xff,0xc5,0x30,0x6e]
// CHECK: fmaxnmp v7.2d, v8.2d, v25.2d   // encoding: [0x07,0xc5,0x79,0x6e]

//----------------------------------------------------------------------
// Vector minNum Pairwise (Floating Point)
//----------------------------------------------------------------------
         fminnmp v10.4h, v15.4h, v22.4h
         fminnmp v3.8h, v5.8h, v6.8h
         fminnmp v10.2s, v15.2s, v22.2s
         fminnmp v3.4s, v5.4s, v6.4s
         fminnmp v17.2d, v13.2d, v2.2d

// CHECK: fminnmp v10.4h, v15.4h, v22.4h  // encoding: [0xea,0x05,0xd6,0x2e]
// CHECK: fminnmp v3.8h, v5.8h, v6.8h     // encoding: [0xa3,0x04,0xc6,0x6e]
// CHECK: fminnmp v10.2s, v15.2s, v22.2s  // encoding: [0xea,0xc5,0xb6,0x2e]
// CHECK: fminnmp v3.4s, v5.4s, v6.4s     // encoding: [0xa3,0xc4,0xa6,0x6e]
// CHECK: fminnmp v17.2d, v13.2d, v2.2d   // encoding: [0xb1,0xc5,0xe2,0x6e]

