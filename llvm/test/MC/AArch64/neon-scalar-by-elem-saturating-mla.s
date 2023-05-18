// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

//-----------------------------------------------------------------------------
// Signed saturating doubling multiply-add long (scalar, by element)
//-----------------------------------------------------------------------------
    sqdmlal s0, h0, v0.h[0]
    sqdmlal s7, h1, v4.h[3]
    sqdmlal s11, h16, v8.h[4]
    sqdmlal s30, h30, v15.h[7]
    sqdmlal d0, s0, v3.s[0]
    sqdmlal d30, s30, v30.s[3]
    sqdmlal d8, s9, v14.s[1]

// CHECK: sqdmlal s0, h0, v0.h[0]       // encoding: [0x00,0x30,0x40,0x5f]
// CHECK: sqdmlal s7, h1, v4.h[3]       // encoding: [0x27,0x30,0x74,0x5f]
// CHECK: sqdmlal s11, h16, v8.h[4]     // encoding: [0x0b,0x3a,0x48,0x5f]
// CHECK: sqdmlal s30, h30, v15.h[7]    // encoding: [0xde,0x3b,0x7f,0x5f]
// CHECK: sqdmlal d0, s0, v3.s[0]       // encoding: [0x00,0x30,0x83,0x5f]
// CHECK: sqdmlal d30, s30, v30.s[3]    // encoding: [0xde,0x3b,0xbe,0x5f]
// CHECK: sqdmlal d8, s9, v14.s[1]      // encoding: [0x28,0x31,0xae,0x5f]
 
//-----------------------------------------------------------------------------
// Signed saturating doubling multiply-subtract long (scalar, by element)
//-----------------------------------------------------------------------------
    sqdmlsl s1, h1, v1.h[0]
    sqdmlsl s8, h2, v5.h[1]
    sqdmlsl s12, h13, v14.h[2]
    sqdmlsl s29, h28, v11.h[7]
    sqdmlsl d1, s1, v13.s[0]
    sqdmlsl d31, s31, v31.s[2]
    sqdmlsl d16, s18, v28.s[3]

// CHECK: sqdmlsl s1, h1, v1.h[0]       // encoding: [0x21,0x70,0x41,0x5f]
// CHECK: sqdmlsl s8, h2, v5.h[1]       // encoding: [0x48,0x70,0x55,0x5f]
// CHECK: sqdmlsl s12, h13, v14.h[2]    // encoding: [0xac,0x71,0x6e,0x5f]
// CHECK: sqdmlsl s29, h28, v11.h[7]    // encoding: [0x9d,0x7b,0x7b,0x5f]
// CHECK: sqdmlsl d1, s1, v13.s[0]      // encoding: [0x21,0x70,0x8d,0x5f]
// CHECK: sqdmlsl d31, s31, v31.s[2]    // encoding: [0xff,0x7b,0x9f,0x5f]
// CHECK: sqdmlsl d16, s18, v28.s[3]    // encoding: [0x50,0x7a,0xbc,0x5f]


        

        
        

