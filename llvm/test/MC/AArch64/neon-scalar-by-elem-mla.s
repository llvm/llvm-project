// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

//------------------------------------------------------------------------------
// Floating Point fused multiply-add (scalar, by element)
//------------------------------------------------------------------------------
    fmla    s0, s1, v1.s[0]
    fmla    s30, s11, v1.s[1]
    fmla    s4, s5, v7.s[2]
    fmla    s16, s22, v16.s[3]
    fmla    d0, d1, v1.d[0]
    fmla    d30, d11, v1.d[1]

// CHECK: fmla    s0, s1, v1.s[0]         // encoding: [0x20,0x10,0x81,0x5f]
// CHECK: fmla    s30, s11, v1.s[1]       // encoding: [0x7e,0x11,0xa1,0x5f]
// CHECK: fmla    s4, s5, v7.s[2]         // encoding: [0xa4,0x18,0x87,0x5f]
// CHECK: fmla    s16, s22, v16.s[3]      // encoding: [0xd0,0x1a,0xb0,0x5f]
// CHECK: fmla    d0, d1, v1.d[0]         // encoding: [0x20,0x10,0xc1,0x5f]
// CHECK: fmla    d30, d11, v1.d[1]       // encoding: [0x7e,0x19,0xc1,0x5f]
 
//------------------------------------------------------------------------------
// Floating Point fused multiply-subtract (scalar, by element)
//------------------------------------------------------------------------------

    fmls    s2, s3, v4.s[0]
    fmls    s29, s10, v28.s[1]      
    fmls    s5, s12, v23.s[2]       
    fmls    s7, s17, v26.s[3]       
    fmls    d0, d1, v1.d[0]         
    fmls    d30, d11, v1.d[1]       

// CHECK: fmls    s2, s3, v4.s[0]     // encoding: [0x62,0x50,0x84,0x5f]
// CHECK: fmls    s29, s10, v28.s[1]  // encoding: [0x5d,0x51,0xbc,0x5f]
// CHECK: fmls    s5, s12, v23.s[2]   // encoding: [0x85,0x59,0x97,0x5f]
// CHECK: fmls    s7, s17, v26.s[3]   // encoding: [0x27,0x5a,0xba,0x5f]
// CHECK: fmls    d0, d1, v1.d[0]     // encoding: [0x20,0x50,0xc1,0x5f]
// CHECK: fmls    d30, d11, v1.d[1]   // encoding: [0x7e,0x59,0xc1,0x5f]

        
        
        

        
        

