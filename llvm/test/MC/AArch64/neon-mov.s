// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64


//----------------------------------------------------------------------
// Vector Move Immediate Shifted
//----------------------------------------------------------------------
         movi v0.2s, #1
         movi v1.2s, #0
         movi v15.2s, #1, lsl #8
         movi v16.2s, #1, lsl #16
         movi v31.2s, #1, lsl #24
         movi v0.4s, #1
         movi v0.4s, #1, lsl #8
         movi v0.4s, #1, lsl #16
         movi v0.4s, #1, lsl #24
         movi v0.4h, #1
         movi v0.4h, #1, lsl #8
         movi v0.8h, #1
         movi v0.8h, #1, lsl #8

// CHECK:  movi v0.2s, #{{0x1|1}}           // encoding: [0x20,0x04,0x00,0x0f]
// CHECK:  movi v1.2s, #{{0x0|0}}           // encoding: [0x01,0x04,0x00,0x0f]
// CHECK:  movi v15.2s, #{{0x1|1}}, lsl #8  // encoding: [0x2f,0x24,0x00,0x0f]
// CHECK:  movi v16.2s, #{{0x1|1}}, lsl #16 // encoding: [0x30,0x44,0x00,0x0f]
// CHECK:  movi v31.2s, #{{0x1|1}}, lsl #24 // encoding: [0x3f,0x64,0x00,0x0f]
// CHECK:  movi v0.4s, #{{0x1|1}}           // encoding: [0x20,0x04,0x00,0x4f]
// CHECK:  movi v0.4s, #{{0x1|1}}, lsl #8   // encoding: [0x20,0x24,0x00,0x4f]
// CHECK:  movi v0.4s, #{{0x1|1}}, lsl #16  // encoding: [0x20,0x44,0x00,0x4f]
// CHECK:  movi v0.4s, #{{0x1|1}}, lsl #24  // encoding: [0x20,0x64,0x00,0x4f]
// CHECK:  movi v0.4h, #{{0x1|1}}           // encoding: [0x20,0x84,0x00,0x0f]
// CHECK:  movi v0.4h, #{{0x1|1}}, lsl #8   // encoding: [0x20,0xa4,0x00,0x0f]
// CHECK:  movi v0.8h, #{{0x1|1}}           // encoding: [0x20,0x84,0x00,0x4f]
// CHECK:  movi v0.8h, #{{0x1|1}}, lsl #8   // encoding: [0x20,0xa4,0x00,0x4f]

//----------------------------------------------------------------------
// Vector Move Inverted Immediate Shifted
//----------------------------------------------------------------------
         mvni v0.2s, #1
         mvni v1.2s, #0
         mvni v0.2s, #1, lsl #8
         mvni v0.2s, #1, lsl #16
         mvni v0.2s, #1, lsl #24
         mvni v0.4s, #1
         mvni v15.4s, #1, lsl #8
         mvni v16.4s, #1, lsl #16
         mvni v31.4s, #1, lsl #24
         mvni v0.4h, #1
         mvni v0.4h, #1, lsl #8
         mvni v0.8h, #1
         mvni v0.8h, #1, lsl #8

// CHECK:  mvni v0.2s, #{{0x1|1}}           // encoding: [0x20,0x04,0x00,0x2f]
// CHECK:  mvni v1.2s, #{{0x0|0}}           // encoding: [0x01,0x04,0x00,0x2f]
// CHECK:  mvni v0.2s, #{{0x1|1}}, lsl #8   // encoding: [0x20,0x24,0x00,0x2f]
// CHECK:  mvni v0.2s, #{{0x1|1}}, lsl #16  // encoding: [0x20,0x44,0x00,0x2f]
// CHECK:  mvni v0.2s, #{{0x1|1}}, lsl #24  // encoding: [0x20,0x64,0x00,0x2f]
// CHECK:  mvni v0.4s, #{{0x1|1}}           // encoding: [0x20,0x04,0x00,0x6f]
// CHECK:  mvni v15.4s, #{{0x1|1}}, lsl #8  // encoding: [0x2f,0x24,0x00,0x6f]
// CHECK:  mvni v16.4s, #{{0x1|1}}, lsl #16 // encoding: [0x30,0x44,0x00,0x6f]
// CHECK:  mvni v31.4s, #{{0x1|1}}, lsl #24 // encoding: [0x3f,0x64,0x00,0x6f]
// CHECK:  mvni v0.4h, #{{0x1|1}}           // encoding: [0x20,0x84,0x00,0x2f]
// CHECK:  mvni v0.4h, #{{0x1|1}}, lsl #8   // encoding: [0x20,0xa4,0x00,0x2f]
// CHECK:  mvni v0.8h, #{{0x1|1}}           // encoding: [0x20,0x84,0x00,0x6f]
// CHECK:  mvni v0.8h, #{{0x1|1}}, lsl #8   // encoding: [0x20,0xa4,0x00,0x6f]

//----------------------------------------------------------------------
// Vector Bitwise Bit Clear (AND NOT) - immediate
//----------------------------------------------------------------------
         bic v0.2s, #1
         bic v1.2s, #0
         bic v0.2s, #1, lsl #8
         bic v0.2s, #1, lsl #16
         bic v0.2s, #1, lsl #24
         bic v0.4s, #1
         bic v0.4s, #1, lsl #8
         bic v0.4s, #1, lsl #16
         bic v0.4s, #1, lsl #24
         bic v15.4h, #1
         bic v16.4h, #1, lsl #8
         bic v0.8h, #1
         bic v31.8h, #1, lsl #8

// CHECK:  bic v0.2s, #{{0x1|1}}           // encoding: [0x20,0x14,0x00,0x2f]
// CHECK:  bic v1.2s, #{{0x0|0}}           // encoding: [0x01,0x14,0x00,0x2f]
// CHECK:  bic v0.2s, #{{0x1|1}}, lsl #8   // encoding: [0x20,0x34,0x00,0x2f]
// CHECK:  bic v0.2s, #{{0x1|1}}, lsl #16  // encoding: [0x20,0x54,0x00,0x2f]
// CHECK:  bic v0.2s, #{{0x1|1}}, lsl #24  // encoding: [0x20,0x74,0x00,0x2f]
// CHECK:  bic v0.4s, #{{0x1|1}}           // encoding: [0x20,0x14,0x00,0x6f]
// CHECK:  bic v0.4s, #{{0x1|1}}, lsl #8   // encoding: [0x20,0x34,0x00,0x6f]
// CHECK:  bic v0.4s, #{{0x1|1}}, lsl #16  // encoding: [0x20,0x54,0x00,0x6f]
// CHECK:  bic v0.4s, #{{0x1|1}}, lsl #24  // encoding: [0x20,0x74,0x00,0x6f]
// CHECK:  bic v15.4h, #{{0x1|1}}          // encoding: [0x2f,0x94,0x00,0x2f]
// CHECK:  bic v16.4h, #{{0x1|1}}, lsl #8  // encoding: [0x30,0xb4,0x00,0x2f]
// CHECK:  bic v0.8h, #{{0x1|1}}           // encoding: [0x20,0x94,0x00,0x6f]
// CHECK:  bic v31.8h, #{{0x1|1}}, lsl #8  // encoding: [0x3f,0xb4,0x00,0x6f]

//----------------------------------------------------------------------
// Vector Bitwise OR - immedidate
//----------------------------------------------------------------------
         orr v0.2s, #1
         orr v1.2s, #0
         orr v0.2s, #1, lsl #8
         orr v0.2s, #1, lsl #16
         orr v0.2s, #1, lsl #24
         orr v0.4s, #1
         orr v0.4s, #1, lsl #8
         orr v0.4s, #1, lsl #16
         orr v0.4s, #1, lsl #24
         orr v31.4h, #1
         orr v15.4h, #1, lsl #8
         orr v0.8h, #1
         orr v16.8h, #1, lsl #8

// CHECK:  orr v0.2s, #{{0x1|1}}           // encoding: [0x20,0x14,0x00,0x0f]
// CHECK:  orr v1.2s, #{{0x0|0}}           // encoding: [0x01,0x14,0x00,0x0f]
// CHECK:  orr v0.2s, #{{0x1|1}}, lsl #8   // encoding: [0x20,0x34,0x00,0x0f]
// CHECK:  orr v0.2s, #{{0x1|1}}, lsl #16  // encoding: [0x20,0x54,0x00,0x0f]
// CHECK:  orr v0.2s, #{{0x1|1}}, lsl #24  // encoding: [0x20,0x74,0x00,0x0f]
// CHECK:  orr v0.4s, #{{0x1|1}}           // encoding: [0x20,0x14,0x00,0x4f]
// CHECK:  orr v0.4s, #{{0x1|1}}, lsl #8   // encoding: [0x20,0x34,0x00,0x4f]
// CHECK:  orr v0.4s, #{{0x1|1}}, lsl #16  // encoding: [0x20,0x54,0x00,0x4f]
// CHECK:  orr v0.4s, #{{0x1|1}}, lsl #24  // encoding: [0x20,0x74,0x00,0x4f]
// CHECK:  orr v31.4h, #{{0x1|1}}          // encoding: [0x3f,0x94,0x00,0x0f]
// CHECK:  orr v15.4h, #{{0x1|1}}, lsl #8  // encoding: [0x2f,0xb4,0x00,0x0f]
// CHECK:  orr v0.8h, #{{0x1|1}}           // encoding: [0x20,0x94,0x00,0x4f]
// CHECK:  orr v16.8h, #{{0x1|1}}, lsl #8  // encoding: [0x30,0xb4,0x00,0x4f]

//----------------------------------------------------------------------
// Vector Move Immediate Masked
//----------------------------------------------------------------------
         movi v0.2s, #1, msl #8
         movi v1.2s, #1, msl #16
         movi v0.4s, #1, msl #8
         movi v31.4s, #1, msl #16

// CHECK:  movi v0.2s, #{{0x1|1}}, msl #8   // encoding: [0x20,0xc4,0x00,0x0f]
// CHECK:  movi v1.2s, #{{0x1|1}}, msl #16  // encoding: [0x21,0xd4,0x00,0x0f]
// CHECK:  movi v0.4s, #{{0x1|1}}, msl #8   // encoding: [0x20,0xc4,0x00,0x4f]
// CHECK:  movi v31.4s, #{{0x1|1}}, msl #16 // encoding: [0x3f,0xd4,0x00,0x4f]

//----------------------------------------------------------------------
// Vector Move Inverted Immediate Masked
//----------------------------------------------------------------------
         mvni v1.2s, #0x1, msl #8
         mvni v0.2s, #0x1, msl #16
         mvni v31.4s, #0x1, msl #8
         mvni v0.4s, #0x1, msl #16

// CHECK:   mvni v1.2s, #{{0x1|1}}, msl #8  // encoding: [0x21,0xc4,0x00,0x2f]
// CHECK:   mvni v0.2s, #{{0x1|1}}, msl #16 // encoding: [0x20,0xd4,0x00,0x2f]
// CHECK:   mvni v31.4s, #{{0x1|1}}, msl #8 // encoding: [0x3f,0xc4,0x00,0x6f]
// CHECK:   mvni v0.4s, #{{0x1|1}}, msl #16 // encoding: [0x20,0xd4,0x00,0x6f]

//----------------------------------------------------------------------
// Vector Immediate - per byte
//----------------------------------------------------------------------
         movi v0.8b, #0
         movi v31.8b, #0xff
         movi v15.16b, #0xf
         movi v31.16b, #0x1f

// CHECK:   movi v0.8b, #{{0x0|0}}        // encoding: [0x00,0xe4,0x00,0x0f]
// CHECK:   movi v31.8b, #{{0xff|255}}      // encoding: [0xff,0xe7,0x07,0x0f]
// CHECK:   movi v15.16b, #{{0xf|15}}      // encoding: [0xef,0xe5,0x00,0x4f]
// CHECK:   movi v31.16b, #{{0x1f|31}}     // encoding: [0xff,0xe7,0x00,0x4f]

//----------------------------------------------------------------------
// Vector Move Immediate - bytemask, per doubleword
//---------------------------------------------------------------------
         movi v0.2d, #0xff00ff00ff00ff00

// CHECK: movi v0.2d, #0xff00ff00ff00ff00 // encoding: [0x40,0xe5,0x05,0x6f]

//----------------------------------------------------------------------
// Vector Move Immediate - bytemask, one doubleword
//----------------------------------------------------------------------
         movi d0, #0xff00ff00ff00ff00

// CHECK: movi d0,  #0xff00ff00ff00ff00 // encoding: [0x40,0xe5,0x05,0x2f]

//----------------------------------------------------------------------
// Vector Floating Point Move Immediate
//----------------------------------------------------------------------
         fmov v1.2s, #1.0
         fmov v15.4s, #1.0
         fmov v31.2d, #1.0

// CHECK:  fmov v1.2s, #{{1.00000000|1.000000e\+00}}     // encoding: [0x01,0xf6,0x03,0x0f]
// CHECK:  fmov v15.4s, #{{1.00000000|1.000000e\+00}}    // encoding: [0x0f,0xf6,0x03,0x4f]
// CHECK:  fmov v31.2d, #{{1.00000000|1.000000e\+00}}    // encoding: [0x1f,0xf6,0x03,0x6f]


//----------------------------------------------------------------------
// Vector Move -  register
//----------------------------------------------------------------------

      mov v0.8b, v31.8b
      mov v15.16b, v16.16b
      orr v0.8b, v31.8b, v31.8b
      orr v15.16b, v16.16b, v16.16b

// CHECK:   mov v0.8b, v31.8b      // encoding: [0xe0,0x1f,0xbf,0x0e]
// CHECK:   mov v15.16b, v16.16b  // encoding: [0x0f,0x1e,0xb0,0x4e]
// CHECK:   mov v0.8b, v31.8b      // encoding: [0xe0,0x1f,0xbf,0x0e]
// CHECK:   mov v15.16b, v16.16b  // encoding: [0x0f,0x1e,0xb0,0x4e]

