// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

//------------------------------------------------------------------------------
// Duplicate element (scalar)
//------------------------------------------------------------------------------
         dup b0, v0.b[15]
         dup b1, v0.b[7]
         dup b17, v0.b[0]
         dup h5, v31.h[7]
         dup h9, v1.h[4]
         dup h11, v17.h[0]
         dup s2, v2.s[3]
         dup s4, v21.s[0]
         dup s31, v21.s[2]
         dup d3, v5.d[0]
         dup d6, v5.d[1]

// CHECK: {{dup|mov}} b0, v0.b[15]      // encoding: [0x00,0x04,0x1f,0x5e]
// CHECK: {{dup|mov}} b1, v0.b[7]       // encoding: [0x01,0x04,0x0f,0x5e]
// CHECK: {{dup|mov}} b17, v0.b[0]      // encoding: [0x11,0x04,0x01,0x5e]
// CHECK: {{dup|mov}} h5, v31.h[7]      // encoding: [0xe5,0x07,0x1e,0x5e]
// CHECK: {{dup|mov}} h9, v1.h[4]       // encoding: [0x29,0x04,0x12,0x5e]
// CHECK: {{dup|mov}} h11, v17.h[0]     // encoding: [0x2b,0x06,0x02,0x5e]
// CHECK: {{dup|mov}} s2, v2.s[3]       // encoding: [0x42,0x04,0x1c,0x5e]
// CHECK: {{dup|mov}} s4, v21.s[0]      // encoding: [0xa4,0x06,0x04,0x5e]
// CHECK: {{dup|mov}} s31, v21.s[2]     // encoding: [0xbf,0x06,0x14,0x5e]
// CHECK: {{dup|mov}} d3, v5.d[0]       // encoding: [0xa3,0x04,0x08,0x5e]
// CHECK: {{dup|mov}} d6, v5.d[1]       // encoding: [0xa6,0x04,0x18,0x5e]

//------------------------------------------------------------------------------
// Aliases for Duplicate element (scalar)
//------------------------------------------------------------------------------
         mov b0, v0.b[15]
         mov b1, v0.b[7]
         mov b17, v0.b[0]
         mov h5, v31.h[7]
         mov h9, v1.h[4]
         mov h11, v17.h[0]
         mov s2, v2.s[3]
         mov s4, v21.s[0]
         mov s31, v21.s[2]
         mov d3, v5.d[0]
         mov d6, v5.d[1]

// CHECK: {{dup|mov}} b0, v0.b[15]      // encoding: [0x00,0x04,0x1f,0x5e]
// CHECK: {{dup|mov}} b1, v0.b[7]       // encoding: [0x01,0x04,0x0f,0x5e]
// CHECK: {{dup|mov}} b17, v0.b[0]      // encoding: [0x11,0x04,0x01,0x5e]
// CHECK: {{dup|mov}} h5, v31.h[7]      // encoding: [0xe5,0x07,0x1e,0x5e]
// CHECK: {{dup|mov}} h9, v1.h[4]       // encoding: [0x29,0x04,0x12,0x5e]
// CHECK: {{dup|mov}} h11, v17.h[0]     // encoding: [0x2b,0x06,0x02,0x5e]
// CHECK: {{dup|mov}} s2, v2.s[3]       // encoding: [0x42,0x04,0x1c,0x5e]
// CHECK: {{dup|mov}} s4, v21.s[0]      // encoding: [0xa4,0x06,0x04,0x5e]
// CHECK: {{dup|mov}} s31, v21.s[2]     // encoding: [0xbf,0x06,0x14,0x5e]
// CHECK: {{dup|mov}} d3, v5.d[0]       // encoding: [0xa3,0x04,0x08,0x5e]
// CHECK: {{dup|mov}} d6, v5.d[1]       // encoding: [0xa6,0x04,0x18,0x5e]
