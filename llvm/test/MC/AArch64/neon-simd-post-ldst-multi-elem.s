// RUN: llvm-mc -triple=arm64 -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//------------------------------------------------------------------------------
// Load multiple 1-element structures from one register (post-index)
//------------------------------------------------------------------------------
         ld1 { v0.16b }, [x0], x1
         ld1 { v15.8h }, [x15], x2
         ld1 { v31.4s }, [sp], #16
         ld1 { v0.2d }, [x0], #16
         ld1 { v0.8b }, [x0], x2
         ld1 { v15.4h }, [x15], x3
         ld1 { v31.2s }, [sp], #8
         ld1 { v0.1d }, [x0], #8
// CHECK: ld1 { v0.16b }, [x0], x1
// CHECK:     // encoding: [0x00,0x70,0xc1,0x4c]
// CHECK: ld1 { v15.8h }, [x15], x2
// CHECK:     // encoding: [0xef,0x75,0xc2,0x4c]
// CHECK: ld1 { v31.4s }, [sp], #16
// CHECK:     // encoding: [0xff,0x7b,0xdf,0x4c]
// CHECK: ld1 { v0.2d }, [x0], #16
// CHECK:     // encoding: [0x00,0x7c,0xdf,0x4c]
// CHECK: ld1 { v0.8b }, [x0], x2
// CHECK:     // encoding: [0x00,0x70,0xc2,0x0c]
// CHECK: ld1 { v15.4h }, [x15], x3
// CHECK:     // encoding: [0xef,0x75,0xc3,0x0c]
// CHECK: ld1 { v31.2s }, [sp], #8
// CHECK:     // encoding: [0xff,0x7b,0xdf,0x0c]
// CHECK: ld1 { v0.1d }, [x0], #8
// CHECK:     // encoding: [0x00,0x7c,0xdf,0x0c]

//------------------------------------------------------------------------------
// Load multiple 1-element structures from two consecutive registers
// (post-index)
//------------------------------------------------------------------------------
         ld1 { v0.16b, v1.16b }, [x0], x1
         ld1 { v15.8h, v16.8h }, [x15], x2
         ld1 { v31.4s, v0.4s }, [sp], #32
         ld1 { v0.2d, v1.2d }, [x0], #32
         ld1 { v0.8b, v1.8b }, [x0], x2
         ld1 { v15.4h, v16.4h }, [x15], x3
         ld1 { v31.2s, v0.2s }, [sp], #16
         ld1 { v0.1d, v1.1d }, [x0], #16
// CHECK: ld1 { v0.16b, v1.16b }, [x0], x1
// CHECK:     // encoding: [0x00,0xa0,0xc1,0x4c]
// CHECK: ld1 { v15.8h, v16.8h }, [x15], x2
// CHECK:     // encoding: [0xef,0xa5,0xc2,0x4c]
// CHECK: ld1 { v31.4s, v0.4s }, [sp], #32
// CHECK:     // encoding: [0xff,0xab,0xdf,0x4c]
// CHECK: ld1 { v0.2d, v1.2d }, [x0], #32
// CHECK:     // encoding: [0x00,0xac,0xdf,0x4c]
// CHECK: ld1 { v0.8b, v1.8b }, [x0], x2
// CHECK:     // encoding: [0x00,0xa0,0xc2,0x0c]
// CHECK: ld1 { v15.4h, v16.4h }, [x15], x3
// CHECK:     // encoding: [0xef,0xa5,0xc3,0x0c]
// CHECK: ld1 { v31.2s, v0.2s }, [sp], #16
// CHECK:     // encoding: [0xff,0xab,0xdf,0x0c]
// CHECK: ld1 { v0.1d, v1.1d }, [x0], #16
// CHECK:     // encoding: [0x00,0xac,0xdf,0x0c]

//------------------------------------------------------------------------------
// Load multiple 1-element structures from three consecutive registers
// (post-index)
//------------------------------------------------------------------------------
         ld1 { v0.16b, v1.16b, v2.16b }, [x0], x1
         ld1 { v15.8h, v16.8h, v17.8h }, [x15], x2
         ld1 { v31.4s, v0.4s, v1.4s }, [sp], #48
         ld1 { v0.2d, v1.2d, v2.2d }, [x0], #48
         ld1 { v0.8b, v1.8b, v2.8b }, [x0], x2
         ld1 { v15.4h, v16.4h, v17.4h }, [x15], x3
         ld1 { v31.2s, v0.2s, v1.2s }, [sp], #24
         ld1 { v0.1d, v1.1d, v2.1d }, [x0], #24
// CHECK: ld1 { v0.16b, v1.16b, v2.16b }, [x0], x1
// CHECK:     // encoding: [0x00,0x60,0xc1,0x4c]
// CHECK: ld1 { v15.8h, v16.8h, v17.8h }, [x15], x2
// CHECK:     // encoding: [0xef,0x65,0xc2,0x4c]
// CHECK: ld1 { v31.4s, v0.4s, v1.4s }, [sp], #48
// CHECK:     // encoding: [0xff,0x6b,0xdf,0x4c]
// CHECK: ld1 { v0.2d, v1.2d, v2.2d }, [x0], #48
// CHECK:     // encoding: [0x00,0x6c,0xdf,0x4c]
// CHECK: ld1 { v0.8b, v1.8b, v2.8b }, [x0], x2
// CHECK:     // encoding: [0x00,0x60,0xc2,0x0c]
// CHECK: ld1 { v15.4h, v16.4h, v17.4h }, [x15], x3
// CHECK:     // encoding: [0xef,0x65,0xc3,0x0c]
// CHECK: ld1 { v31.2s, v0.2s, v1.2s }, [sp], #24
// CHECK:     // encoding: [0xff,0x6b,0xdf,0x0c]
// CHECK: ld1 { v0.1d, v1.1d, v2.1d }, [x0], #24
// CHECK:     // encoding: [0x00,0x6c,0xdf,0x0c]

//------------------------------------------------------------------------------
// Load multiple 1-element structures from four consecutive registers
// (post-index)
//------------------------------------------------------------------------------
         ld1 { v0.16b, v1.16b, v2.16b, v3.16b }, [x0], x1
         ld1 { v15.8h, v16.8h, v17.8h, v18.8h }, [x15], x2
         ld1 { v31.4s, v0.4s, v1.4s, v2.4s }, [sp], #64
         ld1 { v0.2d, v1.2d, v2.2d, v3.2d }, [x0], #64
         ld1 { v0.8b, v1.8b, v2.8b, v3.8b }, [x0], x3
         ld1 { v15.4h, v16.4h, v17.4h, v18.4h }, [x15], x4
         ld1 { v31.2s, v0.2s, v1.2s, v2.2s }, [sp], #32
         ld1 { v0.1d, v1.1d, v2.1d, v3.1d }, [x0], #32
// CHECK: ld1 { v0.16b, v1.16b, v2.16b, v3.16b }, [x0], x1
// CHECK:     // encoding: [0x00,0x20,0xc1,0x4c]
// CHECK: ld1 { v15.8h, v16.8h, v17.8h, v18.8h }, [x15], x2
// CHECK:     // encoding: [0xef,0x25,0xc2,0x4c]
// CHECK: ld1 { v31.4s, v0.4s, v1.4s, v2.4s }, [sp], #64
// CHECK:     // encoding: [0xff,0x2b,0xdf,0x4c]
// CHECK: ld1 { v0.2d, v1.2d, v2.2d, v3.2d }, [x0], #64
// CHECK:     // encoding: [0x00,0x2c,0xdf,0x4c]
// CHECK: ld1 { v0.8b, v1.8b, v2.8b, v3.8b }, [x0], x3
// CHECK:     // encoding: [0x00,0x20,0xc3,0x0c]
// CHECK: ld1 { v15.4h, v16.4h, v17.4h, v18.4h }, [x15], x4
// CHECK:     // encoding: [0xef,0x25,0xc4,0x0c]
// CHECK: ld1 { v31.2s, v0.2s, v1.2s, v2.2s }, [sp], #32
// CHECK:     // encoding: [0xff,0x2b,0xdf,0x0c]
// CHECK: ld1 { v0.1d, v1.1d, v2.1d, v3.1d }, [x0], #32
// CHECK:     // encoding: [0x00,0x2c,0xdf,0x0c]

//------------------------------------------------------------------------------
// Load multiple 2-element structures from two consecutive registers
// (post-index)
//------------------------------------------------------------------------------
         ld2 { v0.16b, v1.16b }, [x0], x1
         ld2 { v15.8h, v16.8h }, [x15], x2
         ld2 { v31.4s, v0.4s }, [sp], #32
         ld2 { v0.2d, v1.2d }, [x0], #32
         ld2 { v0.8b, v1.8b }, [x0], x2
         ld2 { v15.4h, v16.4h }, [x15], x3
         ld2 { v31.2s, v0.2s }, [sp], #16
// CHECK: ld2 { v0.16b, v1.16b }, [x0], x1
// CHECK:     // encoding: [0x00,0x80,0xc1,0x4c]
// CHECK: ld2 { v15.8h, v16.8h }, [x15], x2
// CHECK:     // encoding: [0xef,0x85,0xc2,0x4c]
// CHECK: ld2 { v31.4s, v0.4s }, [sp], #32
// CHECK:     // encoding: [0xff,0x8b,0xdf,0x4c]
// CHECK: ld2 { v0.2d, v1.2d }, [x0], #32
// CHECK:     // encoding: [0x00,0x8c,0xdf,0x4c]
// CHECK: ld2 { v0.8b, v1.8b }, [x0], x2
// CHECK:     // encoding: [0x00,0x80,0xc2,0x0c]
// CHECK: ld2 { v15.4h, v16.4h }, [x15], x3
// CHECK:     // encoding: [0xef,0x85,0xc3,0x0c]
// CHECK: ld2 { v31.2s, v0.2s }, [sp], #16
// CHECK:     // encoding: [0xff,0x8b,0xdf,0x0c]

//------------------------------------------------------------------------------
// Load multiple 3-element structures from three consecutive registers
// (post-index)
//------------------------------------------------------------------------------
         ld3 { v0.16b, v1.16b, v2.16b }, [x0], x1
         ld3 { v15.8h, v16.8h, v17.8h }, [x15], x2
         ld3 { v31.4s, v0.4s, v1.4s }, [sp], #48
         ld3 { v0.2d, v1.2d, v2.2d }, [x0], #48
         ld3 { v0.8b, v1.8b, v2.8b }, [x0], x2
         ld3 { v15.4h, v16.4h, v17.4h }, [x15], x3
         ld3 { v31.2s, v0.2s, v1.2s }, [sp], #24
// CHECK: ld3 { v0.16b, v1.16b, v2.16b }, [x0], x1
// CHECK:     // encoding: [0x00,0x40,0xc1,0x4c]
// CHECK: ld3 { v15.8h, v16.8h, v17.8h }, [x15], x2
// CHECK:     // encoding: [0xef,0x45,0xc2,0x4c]
// CHECK: ld3 { v31.4s, v0.4s, v1.4s }, [sp], #48
// CHECK:     // encoding: [0xff,0x4b,0xdf,0x4c]
// CHECK: ld3 { v0.2d, v1.2d, v2.2d }, [x0], #48
// CHECK:     // encoding: [0x00,0x4c,0xdf,0x4c]
// CHECK: ld3 { v0.8b, v1.8b, v2.8b }, [x0], x2
// CHECK:     // encoding: [0x00,0x40,0xc2,0x0c]
// CHECK: ld3 { v15.4h, v16.4h, v17.4h }, [x15], x3
// CHECK:     // encoding: [0xef,0x45,0xc3,0x0c]
// CHECK: ld3 { v31.2s, v0.2s, v1.2s }, [sp], #24
// CHECK:     // encoding: [0xff,0x4b,0xdf,0x0c]

//------------------------------------------------------------------------------
// Load multiple 4-element structures from four consecutive registers
// (post-index)
//------------------------------------------------------------------------------
         ld4 { v0.16b, v1.16b, v2.16b, v3.16b }, [x0], x1
         ld4 { v15.8h, v16.8h, v17.8h, v18.8h }, [x15], x2
         ld4 { v31.4s, v0.4s, v1.4s, v2.4s }, [sp], #64
         ld4 { v0.2d, v1.2d, v2.2d, v3.2d }, [x0], #64
         ld4 { v0.8b, v1.8b, v2.8b, v3.8b }, [x0], x3
         ld4 { v15.4h, v16.4h, v17.4h, v18.4h }, [x15], x4
         ld4 { v31.2s, v0.2s, v1.2s, v2.2s }, [sp], #32
// CHECK: ld4 { v0.16b, v1.16b, v2.16b, v3.16b }, [x0], x1
// CHECK:     // encoding: [0x00,0x00,0xc1,0x4c]
// CHECK: ld4 { v15.8h, v16.8h, v17.8h, v18.8h }, [x15], x2
// CHECK:     // encoding: [0xef,0x05,0xc2,0x4c]
// CHECK: ld4 { v31.4s, v0.4s, v1.4s, v2.4s }, [sp], #64
// CHECK:     // encoding: [0xff,0x0b,0xdf,0x4c]
// CHECK: ld4 { v0.2d, v1.2d, v2.2d, v3.2d }, [x0], #64
// CHECK:     // encoding: [0x00,0x0c,0xdf,0x4c]
// CHECK: ld4 { v0.8b, v1.8b, v2.8b, v3.8b }, [x0], x3
// CHECK:     // encoding: [0x00,0x00,0xc3,0x0c]
// CHECK: ld4 { v15.4h, v16.4h, v17.4h, v18.4h }, [x15], x4
// CHECK:     // encoding: [0xef,0x05,0xc4,0x0c]
// CHECK: ld4 { v31.2s, v0.2s, v1.2s, v2.2s }, [sp], #32
// CHECK:     // encoding: [0xff,0x0b,0xdf,0x0c]

//------------------------------------------------------------------------------
// Store multiple 1-element structures from one register (post-index)
//------------------------------------------------------------------------------
         st1 { v0.16b }, [x0], x1
         st1 { v15.8h }, [x15], x2
         st1 { v31.4s }, [sp], #16
         st1 { v0.2d }, [x0], #16
         st1 { v0.8b }, [x0], x2
         st1 { v15.4h }, [x15], x3
         st1 { v31.2s }, [sp], #8
         st1 { v0.1d }, [x0], #8
// CHECK: st1 { v0.16b }, [x0], x1
// CHECK:     // encoding: [0x00,0x70,0x81,0x4c]
// CHECK: st1 { v15.8h }, [x15], x2
// CHECK:     // encoding: [0xef,0x75,0x82,0x4c]
// CHECK: st1 { v31.4s }, [sp], #16
// CHECK:     // encoding: [0xff,0x7b,0x9f,0x4c]
// CHECK: st1 { v0.2d }, [x0], #16
// CHECK:     // encoding: [0x00,0x7c,0x9f,0x4c]
// CHECK: st1 { v0.8b }, [x0], x2
// CHECK:     // encoding: [0x00,0x70,0x82,0x0c]
// CHECK: st1 { v15.4h }, [x15], x3
// CHECK:     // encoding: [0xef,0x75,0x83,0x0c]
// CHECK: st1 { v31.2s }, [sp], #8
// CHECK:     // encoding: [0xff,0x7b,0x9f,0x0c]
// CHECK: st1 { v0.1d }, [x0], #8
// CHECK:     // encoding: [0x00,0x7c,0x9f,0x0c]

//------------------------------------------------------------------------------
// Store multiple 1-element structures from two consecutive registers
// (post-index)
//------------------------------------------------------------------------------
         st1 { v0.16b, v1.16b }, [x0], x1
         st1 { v15.8h, v16.8h }, [x15], x2
         st1 { v31.4s, v0.4s }, [sp], #32
         st1 { v0.2d, v1.2d }, [x0], #32
         st1 { v0.8b, v1.8b }, [x0], x2
         st1 { v15.4h, v16.4h }, [x15], x3
         st1 { v31.2s, v0.2s }, [sp], #16
         st1 { v0.1d, v1.1d }, [x0], #16
// CHECK: st1 { v0.16b, v1.16b }, [x0], x1
// CHECK:     // encoding: [0x00,0xa0,0x81,0x4c]
// CHECK: st1 { v15.8h, v16.8h }, [x15], x2
// CHECK:     // encoding: [0xef,0xa5,0x82,0x4c]
// CHECK: st1 { v31.4s, v0.4s }, [sp], #32
// CHECK:     // encoding: [0xff,0xab,0x9f,0x4c]
// CHECK: st1 { v0.2d, v1.2d }, [x0], #32
// CHECK:     // encoding: [0x00,0xac,0x9f,0x4c]
// CHECK: st1 { v0.8b, v1.8b }, [x0], x2
// CHECK:     // encoding: [0x00,0xa0,0x82,0x0c]
// CHECK: st1 { v15.4h, v16.4h }, [x15], x3
// CHECK:     // encoding: [0xef,0xa5,0x83,0x0c]
// CHECK: st1 { v31.2s, v0.2s }, [sp], #16
// CHECK:     // encoding: [0xff,0xab,0x9f,0x0c]
// CHECK: st1 { v0.1d, v1.1d }, [x0], #16
// CHECK:     // encoding: [0x00,0xac,0x9f,0x0c]

//------------------------------------------------------------------------------
// Store multiple 1-element structures from three consecutive registers
// (post-index)
//------------------------------------------------------------------------------
         st1 { v0.16b, v1.16b, v2.16b }, [x0], x1
         st1 { v15.8h, v16.8h, v17.8h }, [x15], x2
         st1 { v31.4s, v0.4s, v1.4s }, [sp], #48
         st1 { v0.2d, v1.2d, v2.2d }, [x0], #48
         st1 { v0.8b, v1.8b, v2.8b }, [x0], x2
         st1 { v15.4h, v16.4h, v17.4h }, [x15], x3
         st1 { v31.2s, v0.2s, v1.2s }, [sp], #24
         st1 { v0.1d, v1.1d, v2.1d }, [x0], #24
// CHECK: st1 { v0.16b, v1.16b, v2.16b }, [x0], x1
// CHECK:     // encoding: [0x00,0x60,0x81,0x4c]
// CHECK: st1 { v15.8h, v16.8h, v17.8h }, [x15], x2
// CHECK:     // encoding: [0xef,0x65,0x82,0x4c]
// CHECK: st1 { v31.4s, v0.4s, v1.4s }, [sp], #48
// CHECK:     // encoding: [0xff,0x6b,0x9f,0x4c]
// CHECK: st1 { v0.2d, v1.2d, v2.2d }, [x0], #48
// CHECK:     // encoding: [0x00,0x6c,0x9f,0x4c]
// CHECK: st1 { v0.8b, v1.8b, v2.8b }, [x0], x2
// CHECK:     // encoding: [0x00,0x60,0x82,0x0c]
// CHECK: st1 { v15.4h, v16.4h, v17.4h }, [x15], x3
// CHECK:     // encoding: [0xef,0x65,0x83,0x0c]
// CHECK: st1 { v31.2s, v0.2s, v1.2s }, [sp], #24
// CHECK:     // encoding: [0xff,0x6b,0x9f,0x0c]
// CHECK: st1 { v0.1d, v1.1d, v2.1d }, [x0], #24
// CHECK:     // encoding: [0x00,0x6c,0x9f,0x0c]

//------------------------------------------------------------------------------
// Store multiple 1-element structures from four consecutive registers
// (post-index)
//------------------------------------------------------------------------------
         st1 { v0.16b, v1.16b, v2.16b, v3.16b }, [x0], x1
         st1 { v15.8h, v16.8h, v17.8h, v18.8h }, [x15], x2
         st1 { v31.4s, v0.4s, v1.4s, v2.4s }, [sp], #64
         st1 { v0.2d, v1.2d, v2.2d, v3.2d }, [x0], #64
         st1 { v0.8b, v1.8b, v2.8b, v3.8b }, [x0], x3
         st1 { v15.4h, v16.4h, v17.4h, v18.4h }, [x15], x4
         st1 { v31.2s, v0.2s, v1.2s, v2.2s }, [sp], #32
         st1 { v0.1d, v1.1d, v2.1d, v3.1d }, [x0], #32
// CHECK: st1 { v0.16b, v1.16b, v2.16b, v3.16b }, [x0], x1
// CHECK:     // encoding: [0x00,0x20,0x81,0x4c]
// CHECK: st1 { v15.8h, v16.8h, v17.8h, v18.8h }, [x15], x2
// CHECK:     // encoding: [0xef,0x25,0x82,0x4c]
// CHECK: st1 { v31.4s, v0.4s, v1.4s, v2.4s }, [sp], #64
// CHECK:     // encoding: [0xff,0x2b,0x9f,0x4c]
// CHECK: st1 { v0.2d, v1.2d, v2.2d, v3.2d }, [x0], #64
// CHECK:     // encoding: [0x00,0x2c,0x9f,0x4c]
// CHECK: st1 { v0.8b, v1.8b, v2.8b, v3.8b }, [x0], x3
// CHECK:     // encoding: [0x00,0x20,0x83,0x0c]
// CHECK: st1 { v15.4h, v16.4h, v17.4h, v18.4h }, [x15], x4
// CHECK:     // encoding: [0xef,0x25,0x84,0x0c]
// CHECK: st1 { v31.2s, v0.2s, v1.2s, v2.2s }, [sp], #32
// CHECK:     // encoding: [0xff,0x2b,0x9f,0x0c]
// CHECK: st1 { v0.1d, v1.1d, v2.1d, v3.1d }, [x0], #32
// CHECK:     // encoding: [0x00,0x2c,0x9f,0x0c]

//------------------------------------------------------------------------------
// Store multiple 2-element structures from two consecutive registers
// (post-index)
//------------------------------------------------------------------------------
         st2 { v0.16b, v1.16b }, [x0], x1
         st2 { v15.8h, v16.8h }, [x15], x2
         st2 { v31.4s, v0.4s }, [sp], #32
         st2 { v0.2d, v1.2d }, [x0], #32
         st2 { v0.8b, v1.8b }, [x0], x2
         st2 { v15.4h, v16.4h }, [x15], x3
         st2 { v31.2s, v0.2s }, [sp], #16
// CHECK: st2 { v0.16b, v1.16b }, [x0], x1
// CHECK:     // encoding: [0x00,0x80,0x81,0x4c]
// CHECK: st2 { v15.8h, v16.8h }, [x15], x2
// CHECK:     // encoding: [0xef,0x85,0x82,0x4c]
// CHECK: st2 { v31.4s, v0.4s }, [sp], #32
// CHECK:     // encoding: [0xff,0x8b,0x9f,0x4c]
// CHECK: st2 { v0.2d, v1.2d }, [x0], #32
// CHECK:     // encoding: [0x00,0x8c,0x9f,0x4c]
// CHECK: st2 { v0.8b, v1.8b }, [x0], x2
// CHECK:     // encoding: [0x00,0x80,0x82,0x0c]
// CHECK: st2 { v15.4h, v16.4h }, [x15], x3
// CHECK:     // encoding: [0xef,0x85,0x83,0x0c]
// CHECK: st2 { v31.2s, v0.2s }, [sp], #16
// CHECK:     // encoding: [0xff,0x8b,0x9f,0x0c]

//------------------------------------------------------------------------------
// Store multiple 3-element structures from three consecutive registers
// (post-index)
//------------------------------------------------------------------------------
         st3 { v0.16b, v1.16b, v2.16b }, [x0], x1
         st3 { v15.8h, v16.8h, v17.8h }, [x15], x2
         st3 { v31.4s, v0.4s, v1.4s }, [sp], #48
         st3 { v0.2d, v1.2d, v2.2d }, [x0], #48
         st3 { v0.8b, v1.8b, v2.8b }, [x0], x2
         st3 { v15.4h, v16.4h, v17.4h }, [x15], x3
         st3 { v31.2s, v0.2s, v1.2s }, [sp], #24
// CHECK: st3 { v0.16b, v1.16b, v2.16b }, [x0], x1
// CHECK:     // encoding: [0x00,0x40,0x81,0x4c]
// CHECK: st3 { v15.8h, v16.8h, v17.8h }, [x15], x2
// CHECK:     // encoding: [0xef,0x45,0x82,0x4c]
// CHECK: st3 { v31.4s, v0.4s, v1.4s }, [sp], #48
// CHECK:     // encoding: [0xff,0x4b,0x9f,0x4c]
// CHECK: st3 { v0.2d, v1.2d, v2.2d }, [x0], #48
// CHECK:     // encoding: [0x00,0x4c,0x9f,0x4c]
// CHECK: st3 { v0.8b, v1.8b, v2.8b }, [x0], x2
// CHECK:     // encoding: [0x00,0x40,0x82,0x0c]
// CHECK: st3 { v15.4h, v16.4h, v17.4h }, [x15], x3
// CHECK:     // encoding: [0xef,0x45,0x83,0x0c]
// CHECK: st3 { v31.2s, v0.2s, v1.2s }, [sp], #24
// CHECK:     // encoding: [0xff,0x4b,0x9f,0x0c]

//------------------------------------------------------------------------------
// Store multiple 4-element structures from four consecutive registers
// (post-index)
//------------------------------------------------------------------------------
         st4 { v0.16b, v1.16b, v2.16b, v3.16b }, [x0], x1
         st4 { v15.8h, v16.8h, v17.8h, v18.8h }, [x15], x2
         st4 { v31.4s, v0.4s, v1.4s, v2.4s }, [sp], #64
         st4 { v0.2d, v1.2d, v2.2d, v3.2d }, [x0], #64
         st4 { v0.8b, v1.8b, v2.8b, v3.8b }, [x0], x3
         st4 { v15.4h, v16.4h, v17.4h, v18.4h }, [x15], x4
         st4 { v31.2s, v0.2s, v1.2s, v2.2s }, [sp], #32
// CHECK: st4 { v0.16b, v1.16b, v2.16b, v3.16b }, [x0], x1
// CHECK:     // encoding: [0x00,0x00,0x81,0x4c]
// CHECK: st4 { v15.8h, v16.8h, v17.8h, v18.8h }, [x15], x2
// CHECK:     // encoding: [0xef,0x05,0x82,0x4c]
// CHECK: st4 { v31.4s, v0.4s, v1.4s, v2.4s }, [sp], #64
// CHECK:     // encoding: [0xff,0x0b,0x9f,0x4c]
// CHECK: st4 { v0.2d, v1.2d, v2.2d, v3.2d }, [x0], #64
// CHECK:     // encoding: [0x00,0x0c,0x9f,0x4c]
// CHECK: st4 { v0.8b, v1.8b, v2.8b, v3.8b }, [x0], x3
// CHECK:     // encoding: [0x00,0x00,0x83,0x0c]
// CHECK: st4 { v15.4h, v16.4h, v17.4h, v18.4h }, [x15], x4
// CHECK:     // encoding: [0xef,0x05,0x84,0x0c]
// CHECK: st4 { v31.2s, v0.2s, v1.2s, v2.2s }, [sp], #32
// CHECK:     // encoding: [0xff,0x0b,0x9f,0x0c]
