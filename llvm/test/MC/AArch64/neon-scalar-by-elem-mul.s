// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

//------------------------------------------------------------------------------
// Floating Point  multiply (scalar, by element)
//------------------------------------------------------------------------------
    fmul    s0, s1, v1.s[0]
    fmul    s30, s11, v1.s[1]
    fmul    s4, s5, v7.s[2]
    fmul    s16, s22, v16.s[3]
    fmul    d0, d1, v1.d[0]
    fmul    d30, d11, v1.d[1]

// CHECK: fmul    s0, s1, v1.s[0]      // encoding: [0x20,0x90,0x81,0x5f]
// CHECK: fmul    s30, s11, v1.s[1]    // encoding: [0x7e,0x91,0xa1,0x5f]
// CHECK: fmul    s4, s5, v7.s[2]      // encoding: [0xa4,0x98,0x87,0x5f]
// CHECK: fmul    s16, s22, v16.s[3]   // encoding: [0xd0,0x9a,0xb0,0x5f]
// CHECK: fmul    d0, d1, v1.d[0]      // encoding: [0x20,0x90,0xc1,0x5f]
// CHECK: fmul    d30, d11, v1.d[1]    // encoding: [0x7e,0x99,0xc1,0x5f]


//------------------------------------------------------------------------------
// Floating Point  multiply extended (scalar, by element)
//------------------------------------------------------------------------------
    fmulx   s6, s2, v8.s[0]
    fmulx   s7, s3, v13.s[1]
    fmulx   s9, s7, v9.s[2]
    fmulx   s13, s21, v10.s[3]
    fmulx   d15, d9, v7.d[0]
    fmulx   d13, d12, v11.d[1]

// CHECK: fmulx   s6, s2, v8.s[0]         // encoding: [0x46,0x90,0x88,0x7f]
// CHECK: fmulx   s7, s3, v13.s[1]        // encoding: [0x67,0x90,0xad,0x7f]
// CHECK: fmulx   s9, s7, v9.s[2]         // encoding: [0xe9,0x98,0x89,0x7f]
// CHECK: fmulx   s13, s21, v10.s[3]      // encoding: [0xad,0x9a,0xaa,0x7f]
// CHECK: fmulx   d15, d9, v7.d[0]        // encoding: [0x2f,0x91,0xc7,0x7f]
// CHECK: fmulx   d13, d12, v11.d[1]      // encoding: [0x8d,0x99,0xcb,0x7f]

