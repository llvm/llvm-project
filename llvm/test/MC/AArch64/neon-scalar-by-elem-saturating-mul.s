// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

//-----------------------------------------------------------------------------
// Signed saturating doubling multiply long (scalar, by element)
//-----------------------------------------------------------------------------
    sqdmull s1, h1, v1.h[1]
    sqdmull s8, h2, v5.h[2]
    sqdmull s12, h17, v9.h[3]
    sqdmull s31, h31, v15.h[7]
    sqdmull d1, s1, v4.s[0]
    sqdmull d31, s31, v31.s[3]
    sqdmull d9, s10, v15.s[0]


// CHECK: sqdmull s1, h1, v1.h[1]       // encoding: [0x21,0xb0,0x51,0x5f]
// CHECK: sqdmull s8, h2, v5.h[2]       // encoding: [0x48,0xb0,0x65,0x5f]
// CHECK: sqdmull s12, h17, v9.h[3]     // encoding: [0x2c,0xb2,0x79,0x5f]
// CHECK: sqdmull s31, h31, v15.h[7]    // encoding: [0xff,0xbb,0x7f,0x5f]
// CHECK: sqdmull d1, s1, v4.s[0]       // encoding: [0x21,0xb0,0x84,0x5f]
// CHECK: sqdmull d31, s31, v31.s[3]    // encoding: [0xff,0xbb,0xbf,0x5f]
// CHECK: sqdmull d9, s10, v15.s[0]     // encoding: [0x49,0xb1,0x8f,0x5f]
 
//-----------------------------------------------------------------------------
// Scalar Signed saturating doubling multiply returning
// high half (scalar, by element)
//-----------------------------------------------------------------------------
    sqdmulh h0, h1, v0.h[0]
    sqdmulh h10, h11, v10.h[4]
    sqdmulh h20, h21, v15.h[7]
    sqdmulh s25, s26, v27.s[3]
    sqdmulh s2, s6, v7.s[0]

// CHECK: sqdmulh h0, h1, v0.h[0]       // encoding: [0x20,0xc0,0x40,0x5f]
// CHECK: sqdmulh h10, h11, v10.h[4]    // encoding: [0x6a,0xc9,0x4a,0x5f]
// CHECK: sqdmulh h20, h21, v15.h[7]    // encoding: [0xb4,0xca,0x7f,0x5f]
// CHECK: sqdmulh s25, s26, v27.s[3]    // encoding: [0x59,0xcb,0xbb,0x5f]
// CHECK: sqdmulh s2, s6, v7.s[0]       // encoding: [0xc2,0xc0,0x87,0x5f]

//-----------------------------------------------------------------------------
// Signed saturating rounding doubling multiply returning
// high half (scalar, by element)
//-----------------------------------------------------------------------------
    sqrdmulh h31, h30, v14.h[2]
    sqrdmulh h1, h1, v1.h[4]
    sqrdmulh h21, h22, v15.h[7]
    sqrdmulh s5, s6, v7.s[2]
    sqrdmulh s20, s26, v27.s[1]

// CHECK: sqrdmulh h31, h30, v14.h[2]   // encoding: [0xdf,0xd3,0x6e,0x5f]
// CHECK: sqrdmulh h1, h1, v1.h[4]      // encoding: [0x21,0xd8,0x41,0x5f]
// CHECK: sqrdmulh h21, h22, v15.h[7]   // encoding: [0xd5,0xda,0x7f,0x5f]
// CHECK: sqrdmulh s5, s6, v7.s[2]      // encoding: [0xc5,0xd8,0x87,0x5f]
// CHECK: sqrdmulh s20, s26, v27.s[1]   // encoding: [0x54,0xd3,0xbb,0x5f]
        

        
        

