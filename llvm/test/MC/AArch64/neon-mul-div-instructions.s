// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//----------------------------------------------------------------------
// Vector Integer Mul
//----------------------------------------------------------------------
         mul v0.8b, v1.8b, v2.8b
         mul v0.16b, v1.16b, v2.16b
         mul v0.4h, v1.4h, v2.4h
         mul v0.8h, v1.8h, v2.8h
         mul v0.2s, v1.2s, v2.2s
         mul v0.4s, v1.4s, v2.4s

// CHECK: mul v0.8b, v1.8b, v2.8b        // encoding: [0x20,0x9c,0x22,0x0e]
// CHECK: mul v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x9c,0x22,0x4e]
// CHECK: mul v0.4h, v1.4h, v2.4h        // encoding: [0x20,0x9c,0x62,0x0e]
// CHECK: mul v0.8h, v1.8h, v2.8h        // encoding: [0x20,0x9c,0x62,0x4e]
// CHECK: mul v0.2s, v1.2s, v2.2s        // encoding: [0x20,0x9c,0xa2,0x0e]
// CHECK: mul v0.4s, v1.4s, v2.4s        // encoding: [0x20,0x9c,0xa2,0x4e]


//----------------------------------------------------------------------
// Vector Floating-Point Mul
//----------------------------------------------------------------------
         fmul v0.2s, v1.2s, v2.2s
         fmul v0.4s, v1.4s, v2.4s
         fmul v0.2d, v1.2d, v2.2d

// CHECK: fmul v0.2s, v1.2s, v2.2s       // encoding: [0x20,0xdc,0x22,0x2e]
// CHECK: fmul v0.4s, v1.4s, v2.4s       // encoding: [0x20,0xdc,0x22,0x6e]
// CHECK: fmul v0.2d, v1.2d, v2.2d       // encoding: [0x20,0xdc,0x62,0x6e]

//----------------------------------------------------------------------
// Vector Floating-Point Div
//----------------------------------------------------------------------
         fdiv v0.2s, v1.2s, v2.2s
         fdiv v0.4s, v1.4s, v2.4s
         fdiv v0.2d, v1.2d, v2.2d

// CHECK: fdiv v0.2s, v1.2s, v2.2s       // encoding: [0x20,0xfc,0x22,0x2e]
// CHECK: fdiv v0.4s, v1.4s, v2.4s       // encoding: [0x20,0xfc,0x22,0x6e]
// CHECK: fdiv v0.2d, v1.2d, v2.2d       // encoding: [0x20,0xfc,0x62,0x6e]

//----------------------------------------------------------------------
// Vector Multiply (Polynomial)
//----------------------------------------------------------------------
         pmul v17.8b, v31.8b, v16.8b
         pmul v0.16b, v1.16b, v2.16b

// CHECK: pmul v17.8b, v31.8b, v16.8b     // encoding: [0xf1,0x9f,0x30,0x2e]
// CHECK: pmul v0.16b, v1.16b, v2.16b     // encoding: [0x20,0x9c,0x22,0x6e]

//----------------------------------------------------------------------
// Vector Saturating Doubling Multiply High
//----------------------------------------------------------------------
         sqdmulh v2.4h, v25.4h, v3.4h
         sqdmulh v12.8h, v5.8h, v13.8h
         sqdmulh v3.2s, v1.2s, v30.2s

// CHECK: sqdmulh v2.4h, v25.4h, v3.4h    // encoding: [0x22,0xb7,0x63,0x0e]
// CHECK: sqdmulh v12.8h, v5.8h, v13.8h   // encoding: [0xac,0xb4,0x6d,0x4e]
// CHECK: sqdmulh v3.2s, v1.2s, v30.2s    // encoding: [0x23,0xb4,0xbe,0x0e]

//----------------------------------------------------------------------
// Vector Saturating Rouding Doubling Multiply High
//----------------------------------------------------------------------
         sqrdmulh v2.4h, v25.4h, v3.4h
         sqrdmulh v12.8h, v5.8h, v13.8h
         sqrdmulh v3.2s, v1.2s, v30.2s

// CHECK: sqrdmulh v2.4h, v25.4h, v3.4h    // encoding: [0x22,0xb7,0x63,0x2e]
// CHECK: sqrdmulh v12.8h, v5.8h, v13.8h   // encoding: [0xac,0xb4,0x6d,0x6e]
// CHECK: sqrdmulh v3.2s, v1.2s, v30.2s    // encoding: [0x23,0xb4,0xbe,0x2e]

//----------------------------------------------------------------------
// Vector Multiply Extended
//----------------------------------------------------------------------
      fmulx v21.2s, v5.2s, v13.2s
      fmulx v1.4s, v25.4s, v3.4s
      fmulx v31.2d, v22.2d, v2.2d

// CHECK: fmulx v21.2s, v5.2s, v13.2s // encoding: [0xb5,0xdc,0x2d,0x0e]
// CHECK: fmulx v1.4s, v25.4s, v3.4s // encoding: [0x21,0xdf,0x23,0x4e]
// CHECK: fmulx v31.2d, v22.2d, v2.2d // encoding: [0xdf,0xde,0x62,0x4e]

