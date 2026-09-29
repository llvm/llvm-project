// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon,+fullfp16 -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//----------------------------------------------------------------------
// Vector Compare Mask Equal (Integer)
//----------------------------------------------------------------------

         cmeq v0.8b, v15.8b, v17.8b
         cmeq v1.16b, v31.16b, v8.16b
         cmeq v15.4h, v16.4h, v17.4h
         cmeq v5.8h, v6.8h, v7.8h
         cmeq v29.2s, v27.2s, v28.2s
         cmeq v9.4s, v7.4s, v8.4s
         cmeq v3.2d, v31.2d, v21.2d

// CHECK: cmeq v0.8b, v15.8b, v17.8b    // encoding: [0xe0,0x8d,0x31,0x2e]
// CHECK: cmeq v1.16b, v31.16b, v8.16b  // encoding: [0xe1,0x8f,0x28,0x6e]
// CHECK: cmeq v15.4h, v16.4h, v17.4h   // encoding: [0x0f,0x8e,0x71,0x2e]
// CHECK: cmeq v5.8h, v6.8h, v7.8h      // encoding: [0xc5,0x8c,0x67,0x6e]
// CHECK: cmeq v29.2s, v27.2s, v28.2s   // encoding: [0x7d,0x8f,0xbc,0x2e]
// CHECK: cmeq v9.4s, v7.4s, v8.4s      // encoding: [0xe9,0x8c,0xa8,0x6e]
// CHECK: cmeq v3.2d, v31.2d, v21.2d    // encoding: [0xe3,0x8f,0xf5,0x6e]

//----------------------------------------------------------------------
// Vector Compare Mask Higher or Same (Unsigned Integer)
// Vector Compare Mask Less or Same (Unsigned Integer)
// CMLS is alias for CMHS with operands reversed.
//----------------------------------------------------------------------

         cmhs v0.8b, v15.8b, v17.8b
         cmhs v1.16b, v31.16b, v8.16b
         cmhs v15.4h, v16.4h, v17.4h
         cmhs v5.8h, v6.8h, v7.8h
         cmhs v29.2s, v27.2s, v28.2s
         cmhs v9.4s, v7.4s, v8.4s
         cmhs v3.2d, v31.2d, v21.2d

         cmls v0.8b, v17.8b, v15.8b
         cmls v1.16b, v8.16b, v31.16b
         cmls v15.4h, v17.4h, v16.4h
         cmls v5.8h, v7.8h, v6.8h
         cmls v29.2s, v28.2s, v27.2s
         cmls v9.4s, v8.4s, v7.4s
         cmls v3.2d, v21.2d, v31.2d

// CHECK: cmhs v0.8b, v15.8b, v17.8b   // encoding: [0xe0,0x3d,0x31,0x2e]
// CHECK: cmhs v1.16b, v31.16b, v8.16b // encoding: [0xe1,0x3f,0x28,0x6e]
// CHECK: cmhs v15.4h, v16.4h, v17.4h  // encoding: [0x0f,0x3e,0x71,0x2e]
// CHECK: cmhs v5.8h, v6.8h, v7.8h     // encoding: [0xc5,0x3c,0x67,0x6e]
// CHECK: cmhs v29.2s, v27.2s, v28.2s  // encoding: [0x7d,0x3f,0xbc,0x2e]
// CHECK: cmhs v9.4s, v7.4s, v8.4s     // encoding: [0xe9,0x3c,0xa8,0x6e]
// CHECK: cmhs v3.2d, v31.2d, v21.2d   // encoding: [0xe3,0x3f,0xf5,0x6e]
// CHECK: cmhs v0.8b, v15.8b, v17.8b   // encoding: [0xe0,0x3d,0x31,0x2e]
// CHECK: cmhs v1.16b, v31.16b, v8.16b // encoding: [0xe1,0x3f,0x28,0x6e]
// CHECK: cmhs v15.4h, v16.4h, v17.4h  // encoding: [0x0f,0x3e,0x71,0x2e]
// CHECK: cmhs v5.8h, v6.8h, v7.8h     // encoding: [0xc5,0x3c,0x67,0x6e]
// CHECK: cmhs v29.2s, v27.2s, v28.2s  // encoding: [0x7d,0x3f,0xbc,0x2e]
// CHECK: cmhs v9.4s, v7.4s, v8.4s     // encoding: [0xe9,0x3c,0xa8,0x6e]
// CHECK: cmhs v3.2d, v31.2d, v21.2d   // encoding: [0xe3,0x3f,0xf5,0x6e]

//----------------------------------------------------------------------
// Vector Compare Mask Greater Than or Equal (Integer)
// Vector Compare Mask Less Than or Equal (Integer)
// CMLE is alias for CMGE with operands reversed.
//----------------------------------------------------------------------

         cmge v0.8b, v15.8b, v17.8b
         cmge v1.16b, v31.16b, v8.16b
         cmge v15.4h, v16.4h, v17.4h
         cmge v5.8h, v6.8h, v7.8h
         cmge v29.2s, v27.2s, v28.2s
         cmge v9.4s, v7.4s, v8.4s
         cmge v3.2d, v31.2d, v21.2d

         cmle v0.8b, v17.8b, v15.8b
         cmle v1.16b, v8.16b, v31.16b
         cmle v15.4h, v17.4h, v16.4h
         cmle v5.8h, v7.8h, v6.8h
         cmle v29.2s, v28.2s, v27.2s
         cmle v9.4s, v8.4s, v7.4s
         cmle v3.2d, v21.2d, v31.2d

// CHECK: cmge v0.8b, v15.8b, v17.8b    // encoding: [0xe0,0x3d,0x31,0x0e]
// CHECK: cmge v1.16b, v31.16b, v8.16b  // encoding: [0xe1,0x3f,0x28,0x4e]
// CHECK: cmge v15.4h, v16.4h, v17.4h   // encoding: [0x0f,0x3e,0x71,0x0e]
// CHECK: cmge v5.8h, v6.8h, v7.8h      // encoding: [0xc5,0x3c,0x67,0x4e]
// CHECK: cmge v29.2s, v27.2s, v28.2s   // encoding: [0x7d,0x3f,0xbc,0x0e]
// CHECK: cmge v9.4s, v7.4s, v8.4s      // encoding: [0xe9,0x3c,0xa8,0x4e]
// CHECK: cmge v3.2d, v31.2d, v21.2d    // encoding: [0xe3,0x3f,0xf5,0x4e]
// CHECK: cmge v0.8b, v15.8b, v17.8b    // encoding: [0xe0,0x3d,0x31,0x0e]
// CHECK: cmge v1.16b, v31.16b, v8.16b  // encoding: [0xe1,0x3f,0x28,0x4e]
// CHECK: cmge v15.4h, v16.4h, v17.4h   // encoding: [0x0f,0x3e,0x71,0x0e]
// CHECK: cmge v5.8h, v6.8h, v7.8h      // encoding: [0xc5,0x3c,0x67,0x4e]
// CHECK: cmge v29.2s, v27.2s, v28.2s   // encoding: [0x7d,0x3f,0xbc,0x0e]
// CHECK: cmge v9.4s, v7.4s, v8.4s      // encoding: [0xe9,0x3c,0xa8,0x4e]
// CHECK: cmge v3.2d, v31.2d, v21.2d    // encoding: [0xe3,0x3f,0xf5,0x4e]

//----------------------------------------------------------------------
// Vector Compare Mask Higher (Unsigned Integer)
// Vector Compare Mask Lower (Unsigned Integer)
// CMLO is alias for CMHI with operands reversed.
//----------------------------------------------------------------------

         cmhi v0.8b, v15.8b, v17.8b
         cmhi v1.16b, v31.16b, v8.16b
         cmhi v15.4h, v16.4h, v17.4h
         cmhi v5.8h, v6.8h, v7.8h
         cmhi v29.2s, v27.2s, v28.2s
         cmhi v9.4s, v7.4s, v8.4s
         cmhi v3.2d, v31.2d, v21.2d

         cmlo v0.8b, v17.8b, v15.8b
         cmlo v1.16b, v8.16b, v31.16b
         cmlo v15.4h, v17.4h, v16.4h
         cmlo v5.8h, v7.8h, v6.8h
         cmlo v29.2s, v28.2s, v27.2s
         cmlo v9.4s, v8.4s, v7.4s
         cmlo v3.2d, v21.2d, v31.2d

// CHECK: cmhi v0.8b, v15.8b, v17.8b    // encoding: [0xe0,0x35,0x31,0x2e]
// CHECK: cmhi v1.16b, v31.16b, v8.16b  // encoding: [0xe1,0x37,0x28,0x6e]
// CHECK: cmhi v15.4h, v16.4h, v17.4h   // encoding: [0x0f,0x36,0x71,0x2e]
// CHECK: cmhi v5.8h, v6.8h, v7.8h      // encoding: [0xc5,0x34,0x67,0x6e]
// CHECK: cmhi v29.2s, v27.2s, v28.2s   // encoding: [0x7d,0x37,0xbc,0x2e]
// CHECK: cmhi v9.4s, v7.4s, v8.4s      // encoding: [0xe9,0x34,0xa8,0x6e]
// CHECK: cmhi v3.2d, v31.2d, v21.2d    // encoding: [0xe3,0x37,0xf5,0x6e]
// CHECK: cmhi v0.8b, v15.8b, v17.8b    // encoding: [0xe0,0x35,0x31,0x2e]
// CHECK: cmhi v1.16b, v31.16b, v8.16b  // encoding: [0xe1,0x37,0x28,0x6e]
// CHECK: cmhi v15.4h, v16.4h, v17.4h   // encoding: [0x0f,0x36,0x71,0x2e]
// CHECK: cmhi v5.8h, v6.8h, v7.8h      // encoding: [0xc5,0x34,0x67,0x6e]
// CHECK: cmhi v29.2s, v27.2s, v28.2s   // encoding: [0x7d,0x37,0xbc,0x2e]
// CHECK: cmhi v9.4s, v7.4s, v8.4s      // encoding: [0xe9,0x34,0xa8,0x6e]
// CHECK: cmhi v3.2d, v31.2d, v21.2d    // encoding: [0xe3,0x37,0xf5,0x6e]

//----------------------------------------------------------------------
// Vector Compare Mask Greater Than (Integer)
// Vector Compare Mask Less Than (Integer)
// CMLT is alias for CMGT with operands reversed.
//----------------------------------------------------------------------

         cmgt v0.8b, v15.8b, v17.8b
         cmgt v1.16b, v31.16b, v8.16b
         cmgt v15.4h, v16.4h, v17.4h
         cmgt v5.8h, v6.8h, v7.8h
         cmgt v29.2s, v27.2s, v28.2s
         cmgt v9.4s, v7.4s, v8.4s
         cmgt v3.2d, v31.2d, v21.2d

         cmlt v0.8b, v17.8b, v15.8b
         cmlt v1.16b, v8.16b, v31.16b
         cmlt v15.4h, v17.4h, v16.4h
         cmlt v5.8h, v7.8h, v6.8h
         cmlt v29.2s, v28.2s, v27.2s
         cmlt v9.4s, v8.4s, v7.4s
         cmlt v3.2d, v21.2d, v31.2d

// CHECK: cmgt v0.8b, v15.8b, v17.8b    // encoding: [0xe0,0x35,0x31,0x0e]
// CHECK: cmgt v1.16b, v31.16b, v8.16b  // encoding: [0xe1,0x37,0x28,0x4e]
// CHECK: cmgt v15.4h, v16.4h, v17.4h   // encoding: [0x0f,0x36,0x71,0x0e]
// CHECK: cmgt v5.8h, v6.8h, v7.8h      // encoding: [0xc5,0x34,0x67,0x4e]
// CHECK: cmgt v29.2s, v27.2s, v28.2s   // encoding: [0x7d,0x37,0xbc,0x0e]
// CHECK: cmgt v9.4s, v7.4s, v8.4s      // encoding: [0xe9,0x34,0xa8,0x4e]
// CHECK: cmgt v3.2d, v31.2d, v21.2d    // encoding: [0xe3,0x37,0xf5,0x4e]
// CHECK: cmgt v0.8b, v15.8b, v17.8b    // encoding: [0xe0,0x35,0x31,0x0e]
// CHECK: cmgt v1.16b, v31.16b, v8.16b  // encoding: [0xe1,0x37,0x28,0x4e]
// CHECK: cmgt v15.4h, v16.4h, v17.4h   // encoding: [0x0f,0x36,0x71,0x0e]
// CHECK: cmgt v5.8h, v6.8h, v7.8h      // encoding: [0xc5,0x34,0x67,0x4e]
// CHECK: cmgt v29.2s, v27.2s, v28.2s   // encoding: [0x7d,0x37,0xbc,0x0e]
// CHECK: cmgt v9.4s, v7.4s, v8.4s      // encoding: [0xe9,0x34,0xa8,0x4e]
// CHECK: cmgt v3.2d, v31.2d, v21.2d    // encoding: [0xe3,0x37,0xf5,0x4e]

//----------------------------------------------------------------------
// Vector Compare Mask Bitwise Test (Integer)
//----------------------------------------------------------------------

         cmtst v0.8b, v15.8b, v17.8b
         cmtst v1.16b, v31.16b, v8.16b
         cmtst v15.4h, v16.4h, v17.4h
         cmtst v5.8h, v6.8h, v7.8h
         cmtst v29.2s, v27.2s, v28.2s
         cmtst v9.4s, v7.4s, v8.4s
         cmtst v3.2d, v31.2d, v21.2d

// CHECK: cmtst v0.8b, v15.8b, v17.8b    // encoding: [0xe0,0x8d,0x31,0x0e]
// CHECK: cmtst v1.16b, v31.16b, v8.16b  // encoding: [0xe1,0x8f,0x28,0x4e]
// CHECK: cmtst v15.4h, v16.4h, v17.4h   // encoding: [0x0f,0x8e,0x71,0x0e]
// CHECK: cmtst v5.8h, v6.8h, v7.8h      // encoding: [0xc5,0x8c,0x67,0x4e]
// CHECK: cmtst v29.2s, v27.2s, v28.2s   // encoding: [0x7d,0x8f,0xbc,0x0e]
// CHECK: cmtst v9.4s, v7.4s, v8.4s      // encoding: [0xe9,0x8c,0xa8,0x4e]
// CHECK: cmtst v3.2d, v31.2d, v21.2d    // encoding: [0xe3,0x8f,0xf5,0x4e]

//----------------------------------------------------------------------
// Vector Compare Mask Equal (Floating Point)
//----------------------------------------------------------------------

         fcmeq v0.4h, v31.4h, v16.4h
         fcmeq v4.8h, v7.8h, v15.8h
         fcmeq v0.2s, v31.2s, v16.2s
         fcmeq v4.4s, v7.4s, v15.4s
         fcmeq v29.2d, v2.2d, v5.2d

// CHECK: fcmeq   v0.4h, v31.4h, v16.4h   // encoding: [0xe0,0x27,0x50,0x0e]
// CHECK: fcmeq   v4.8h, v7.8h, v15.8h    // encoding: [0xe4,0x24,0x4f,0x4e]
// CHECK: fcmeq v0.2s, v31.2s, v16.2s // encoding: [0xe0,0xe7,0x30,0x0e]
// CHECK: fcmeq v4.4s, v7.4s, v15.4s  // encoding: [0xe4,0xe4,0x2f,0x4e]
// CHECK: fcmeq v29.2d, v2.2d, v5.2d  // encoding: [0x5d,0xe4,0x65,0x4e]

//----------------------------------------------------------------------
// Vector Compare Mask Greater Than Or Equal (Floating Point)
// Vector Compare Mask Less Than Or Equal (Floating Point)
// FCMLE is alias for FCMGE with operands reversed.
//----------------------------------------------------------------------

         fcmge v3.4h, v8.4h, v12.4h
         fcmge v31.8h, v29.8h, v28.8h
         fcmle v3.4h,  v12.4h, v8.4h
         fcmle v31.8h, v28.8h, v29.8h
         fcmge v31.4s, v29.4s, v28.4s
         fcmge v3.2s, v8.2s, v12.2s
         fcmge v17.2d, v15.2d, v13.2d
         fcmle v31.4s, v28.4s, v29.4s
         fcmle v3.2s,  v12.2s, v8.2s
         fcmle v17.2d, v13.2d, v15.2d

// CHECK: fcmge   v3.4h, v8.4h, v12.4h    // encoding: [0x03,0x25,0x4c,0x2e]
// CHECK: fcmge   v31.8h, v29.8h, v28.8h  // encoding: [0xbf,0x27,0x5c,0x6e]
// CHECK: fcmge   v3.4h, v8.4h, v12.4h    // encoding: [0x03,0x25,0x4c,0x2e]
// CHECK: fcmge   v31.8h, v29.8h, v28.8h  // encoding: [0xbf,0x27,0x5c,0x6e]
// CHECK: fcmge v31.4s, v29.4s, v28.4s  // encoding: [0xbf,0xe7,0x3c,0x6e]
// CHECK: fcmge v3.2s, v8.2s, v12.2s    // encoding: [0x03,0xe5,0x2c,0x2e]
// CHECK: fcmge v17.2d, v15.2d, v13.2d  // encoding: [0xf1,0xe5,0x6d,0x6e]
// CHECK: fcmge v31.4s, v29.4s, v28.4s  // encoding: [0xbf,0xe7,0x3c,0x6e]
// CHECK: fcmge v3.2s,  v8.2s, v12.2s   // encoding: [0x03,0xe5,0x2c,0x2e]
// CHECK: fcmge v17.2d, v15.2d, v13.2d  // encoding: [0xf1,0xe5,0x6d,0x6e]

//----------------------------------------------------------------------
// Vector Compare Mask Greater Than (Floating Point)
// Vector Compare Mask Less Than (Floating Point)
// FCMLT is alias for FCMGT with operands reversed.
//----------------------------------------------------------------------

         fcmgt v0.4h, v31.4h, v16.4h
         fcmgt v4.8h, v7.8h, v15.8h
         fcmlt v0.4h, v16.4h, v31.4h
         fcmlt v4.8h, v15.8h, v7.8h
         fcmgt v0.2s, v31.2s, v16.2s
         fcmgt v4.4s, v7.4s, v15.4s
         fcmgt v29.2d, v2.2d, v5.2d
         fcmlt v0.2s, v16.2s, v31.2s
         fcmlt v4.4s, v15.4s, v7.4s
         fcmlt v29.2d, v5.2d, v2.2d

// CHECK: fcmgt   v0.4h, v31.4h, v16.4h   // encoding: [0xe0,0x27,0xd0,0x2e]
// CHECK: fcmgt   v4.8h, v7.8h, v15.8h    // encoding: [0xe4,0x24,0xcf,0x6e]
// CHECK: fcmgt   v0.4h, v31.4h, v16.4h   // encoding: [0xe0,0x27,0xd0,0x2e]
// CHECK: fcmgt   v4.8h, v7.8h, v15.8h    // encoding: [0xe4,0x24,0xcf,0x6e]
// CHECK: fcmgt v0.2s, v31.2s, v16.2s  // encoding: [0xe0,0xe7,0xb0,0x2e]
// CHECK: fcmgt v4.4s, v7.4s, v15.4s   // encoding: [0xe4,0xe4,0xaf,0x6e]
// CHECK: fcmgt v29.2d, v2.2d, v5.2d   // encoding: [0x5d,0xe4,0xe5,0x6e]
// CHECK: fcmgt v0.2s, v31.2s, v16.2s  // encoding: [0xe0,0xe7,0xb0,0x2e]
// CHECK: fcmgt v4.4s, v7.4s, v15.4s   // encoding: [0xe4,0xe4,0xaf,0x6e]
// CHECK: fcmgt v29.2d, v2.2d, v5.2d   // encoding: [0x5d,0xe4,0xe5,0x6e]


//----------------------------------------------------------------------
// Vector Compare Mask Equal to Zero (Integer)
//----------------------------------------------------------------------

         cmeq v0.8b, v15.8b, #0
         cmeq v1.16b, v31.16b, #0
         cmeq v15.4h, v16.4h, #0
         cmeq v5.8h, v6.8h, #0
         cmeq v29.2s, v27.2s, #0
         cmeq v9.4s, v7.4s, #0
         cmeq v3.2d, v31.2d, #0

// CHECK: cmeq v0.8b, v15.8b, #{{0x0|0}}    // encoding: [0xe0,0x99,0x20,0x0e]
// CHECK: cmeq v1.16b, v31.16b, #{{0x0|0}}  // encoding: [0xe1,0x9b,0x20,0x4e]
// CHECK: cmeq v15.4h, v16.4h, #{{0x0|0}}   // encoding: [0x0f,0x9a,0x60,0x0e]
// CHECK: cmeq v5.8h, v6.8h, #{{0x0|0}}     // encoding: [0xc5,0x98,0x60,0x4e]
// CHECK: cmeq v29.2s, v27.2s, #{{0x0|0}}   // encoding: [0x7d,0x9b,0xa0,0x0e]
// CHECK: cmeq v9.4s, v7.4s, #{{0x0|0}}     // encoding: [0xe9,0x98,0xa0,0x4e]
// CHECK: cmeq v3.2d, v31.2d, #{{0x0|0}}    // encoding: [0xe3,0x9b,0xe0,0x4e]

//----------------------------------------------------------------------
// Vector Compare Mask Greater Than or Equal to Zero (Signed Integer)
//----------------------------------------------------------------------
         cmge v0.8b, v15.8b, #0
         cmge v1.16b, v31.16b, #0
         cmge v15.4h, v16.4h, #0
         cmge v5.8h, v6.8h, #0
         cmge v29.2s, v27.2s, #0
         cmge v17.4s, v20.4s, #0
         cmge v3.2d, v31.2d, #0

// CHECK: cmge v0.8b, v15.8b, #{{0x0|0}}    // encoding: [0xe0,0x89,0x20,0x2e]
// CHECK: cmge v1.16b, v31.16b, #{{0x0|0}}  // encoding: [0xe1,0x8b,0x20,0x6e]
// CHECK: cmge v15.4h, v16.4h, #{{0x0|0}}   // encoding: [0x0f,0x8a,0x60,0x2e]
// CHECK: cmge v5.8h, v6.8h, #{{0x0|0}}     // encoding: [0xc5,0x88,0x60,0x6e]
// CHECK: cmge v29.2s, v27.2s, #{{0x0|0}}   // encoding: [0x7d,0x8b,0xa0,0x2e]
// CHECK: cmge v17.4s, v20.4s, #{{0x0|0}}   // encoding: [0x91,0x8a,0xa0,0x6e]
// CHECK: cmge v3.2d, v31.2d, #{{0x0|0}}    // encoding: [0xe3,0x8b,0xe0,0x6e]

//----------------------------------------------------------------------
// Vector Compare Mask Greater Than Zero (Signed Integer)
//----------------------------------------------------------------------

         cmgt v0.8b, v15.8b, #0
         cmgt v1.16b, v31.16b, #0
         cmgt v15.4h, v16.4h, #0
         cmgt v5.8h, v6.8h, #0
         cmgt v29.2s, v27.2s, #0
         cmgt v9.4s, v7.4s, #0
         cmgt v3.2d, v31.2d, #0

// CHECK: cmgt v0.8b, v15.8b, #{{0x0|0}}    // encoding: [0xe0,0x89,0x20,0x0e]
// CHECK: cmgt v1.16b, v31.16b, #{{0x0|0}}  // encoding: [0xe1,0x8b,0x20,0x4e]
// CHECK: cmgt v15.4h, v16.4h, #{{0x0|0}}   // encoding: [0x0f,0x8a,0x60,0x0e]
// CHECK: cmgt v5.8h, v6.8h, #{{0x0|0}}     // encoding: [0xc5,0x88,0x60,0x4e]
// CHECK: cmgt v29.2s, v27.2s, #{{0x0|0}}   // encoding: [0x7d,0x8b,0xa0,0x0e]
// CHECK: cmgt v9.4s, v7.4s, #{{0x0|0}}     // encoding: [0xe9,0x88,0xa0,0x4e]
// CHECK: cmgt v3.2d, v31.2d, #{{0x0|0}}    // encoding: [0xe3,0x8b,0xe0,0x4e]

//----------------------------------------------------------------------
// Vector Compare Mask Less Than or Equal To Zero (Signed Integer)
//----------------------------------------------------------------------
         cmle v0.8b, v15.8b, #0
         cmle v1.16b, v31.16b, #0
         cmle v15.4h, v16.4h, #0
         cmle v5.8h, v6.8h, #0
         cmle v29.2s, v27.2s, #0
         cmle v9.4s, v7.4s, #0
         cmle v3.2d, v31.2d, #0

// CHECK: cmle v0.8b, v15.8b, #{{0x0|0}}    // encoding: [0xe0,0x99,0x20,0x2e]
// CHECK: cmle v1.16b, v31.16b, #{{0x0|0}}  // encoding: [0xe1,0x9b,0x20,0x6e]
// CHECK: cmle v15.4h, v16.4h, #{{0x0|0}}   // encoding: [0x0f,0x9a,0x60,0x2e]
// CHECK: cmle v5.8h, v6.8h, #{{0x0|0}}     // encoding: [0xc5,0x98,0x60,0x6e]
// CHECK: cmle v29.2s, v27.2s, #{{0x0|0}}   // encoding: [0x7d,0x9b,0xa0,0x2e]
// CHECK: cmle v9.4s, v7.4s, #{{0x0|0}}     // encoding: [0xe9,0x98,0xa0,0x6e]
// CHECK: cmle v3.2d, v31.2d, #{{0x0|0}}    // encoding: [0xe3,0x9b,0xe0,0x6e]

//----------------------------------------------------------------------
// Vector Compare Mask Less Than Zero (Signed Integer)
//----------------------------------------------------------------------
         cmlt v0.8b, v15.8b, #0
         cmlt v1.16b, v31.16b, #0
         cmlt v15.4h, v16.4h, #0
         cmlt v5.8h, v6.8h, #0
         cmlt v29.2s, v27.2s, #0
         cmlt v9.4s, v7.4s, #0
         cmlt v3.2d, v31.2d, #0

// CHECK: cmlt v0.8b, v15.8b, #{{0x0|0}}    // encoding: [0xe0,0xa9,0x20,0x0e]
// CHECK: cmlt v1.16b, v31.16b, #{{0x0|0}}  // encoding: [0xe1,0xab,0x20,0x4e]
// CHECK: cmlt v15.4h, v16.4h, #{{0x0|0}}   // encoding: [0x0f,0xaa,0x60,0x0e]
// CHECK: cmlt v5.8h, v6.8h, #{{0x0|0}}     // encoding: [0xc5,0xa8,0x60,0x4e]
// CHECK: cmlt v29.2s, v27.2s, #{{0x0|0}}   // encoding: [0x7d,0xab,0xa0,0x0e]
// CHECK: cmlt v9.4s, v7.4s, #{{0x0|0}}     // encoding: [0xe9,0xa8,0xa0,0x4e]
// CHECK: cmlt v3.2d, v31.2d, #{{0x0|0}}    // encoding: [0xe3,0xab,0xe0,0x4e]

//----------------------------------------------------------------------
// Vector Compare Mask Equal to Zero (Floating Point)
//----------------------------------------------------------------------
         fcmeq v0.4h, v31.4h, #0.0
         fcmeq v4.8h, v7.8h, #0.0
         fcmeq v0.2s, v31.2s, #0.0
         fcmeq v4.4s, v7.4s, #0.0
         fcmeq v29.2d, v2.2d, #0.0
         fcmeq v0.4h, v31.4h, #0
         fcmeq v4.8h, v7.8h, #0
         fcmeq v0.2s, v31.2s, #0
         fcmeq v4.4s, v7.4s, #0
         fcmeq v29.2d, v2.2d, #0

// CHECK: fcmeq   v0.4h, v31.4h, #0.0     // encoding: [0xe0,0xdb,0xf8,0x0e]
// CHECK: fcmeq   v4.8h, v7.8h, #0.0      // encoding: [0xe4,0xd8,0xf8,0x4e]
// CHECK: fcmeq v0.2s, v31.2s, #0.0  // encoding: [0xe0,0xdb,0xa0,0x0e]
// CHECK: fcmeq v4.4s, v7.4s, #0.0   // encoding: [0xe4,0xd8,0xa0,0x4e]
// CHECK: fcmeq v29.2d, v2.2d, #0.0  // encoding: [0x5d,0xd8,0xe0,0x4e]
// CHECK: fcmeq   v0.4h, v31.4h, #0.0     // encoding: [0xe0,0xdb,0xf8,0x0e]
// CHECK: fcmeq   v4.8h, v7.8h, #0.0      // encoding: [0xe4,0xd8,0xf8,0x4e]
// CHECK: fcmeq v0.2s, v31.2s, #0.0  // encoding: [0xe0,0xdb,0xa0,0x0e]
// CHECK: fcmeq v4.4s, v7.4s, #0.0   // encoding: [0xe4,0xd8,0xa0,0x4e]
// CHECK: fcmeq v29.2d, v2.2d, #0.0  // encoding: [0x5d,0xd8,0xe0,0x4e]

//----------------------------------------------------------------------
// Vector Compare Mask Greater Than or Equal to Zero (Floating Point)
//----------------------------------------------------------------------
         fcmge v3.4h, v8.4h, #0.0
         fcmge v31.8h, v29.8h, #0.0
         fcmge v31.4s, v29.4s, #0.0
         fcmge v3.2s, v8.2s, #0.0
         fcmge v17.2d, v15.2d, #0.0
         fcmge v3.4h, v8.4h, #0
         fcmge v31.8h, v29.8h, #0
         fcmge v31.4s, v29.4s, #0
         fcmge v3.2s, v8.2s, #0
         fcmge v17.2d, v15.2d, #0

// CHECK: fcmge   v3.4h, v8.4h, #0.0      // encoding: [0x03,0xc9,0xf8,0x2e]
// CHECK: fcmge   v31.8h, v29.8h, #0.0    // encoding: [0xbf,0xcb,0xf8,0x6e]
// CHECK: fcmge v31.4s, v29.4s, #0.0  // encoding: [0xbf,0xcb,0xa0,0x6e]
// CHECK: fcmge v3.2s, v8.2s, #0.0    // encoding: [0x03,0xc9,0xa0,0x2e]
// CHECK: fcmge v17.2d, v15.2d, #0.0   // encoding: [0xf1,0xc9,0xe0,0x6e]
// CHECK: fcmge   v3.4h, v8.4h, #0.0      // encoding: [0x03,0xc9,0xf8,0x2e]
// CHECK: fcmge   v31.8h, v29.8h, #0.0    // encoding: [0xbf,0xcb,0xf8,0x6e]
// CHECK: fcmge v31.4s, v29.4s, #0.0  // encoding: [0xbf,0xcb,0xa0,0x6e]
// CHECK: fcmge v3.2s, v8.2s, #0.0    // encoding: [0x03,0xc9,0xa0,0x2e]
// CHECK: fcmge v17.2d, v15.2d, #0.0   // encoding: [0xf1,0xc9,0xe0,0x6e]

//----------------------------------------------------------------------
// Vector Compare Mask Greater Than Zero (Floating Point)
//----------------------------------------------------------------------
         fcmgt v0.4h, v31.4h, #0.0
         fcmgt v4.8h, v7.8h, #0.0
         fcmgt v0.2s, v31.2s, #0.0
         fcmgt v4.4s, v7.4s, #0.0
         fcmgt v29.2d, v2.2d, #0.0
         fcmgt v0.4h, v31.4h, #0
         fcmgt v4.8h, v7.8h, #0
         fcmgt v0.2s, v31.2s, #0
         fcmgt v4.4s, v7.4s, #0
         fcmgt v29.2d, v2.2d, #0

// CHECK: fcmgt   v0.4h, v31.4h, #0.0     // encoding: [0xe0,0xcb,0xf8,0x0e]
// CHECK: fcmgt   v4.8h, v7.8h, #0.0      // encoding: [0xe4,0xc8,0xf8,0x4e]
// CHECK: fcmgt v0.2s, v31.2s, #0.0   // encoding: [0xe0,0xcb,0xa0,0x0e]
// CHECK: fcmgt v4.4s, v7.4s, #0.0    // encoding: [0xe4,0xc8,0xa0,0x4e]
// CHECK: fcmgt v29.2d, v2.2d, #0.0   // encoding: [0x5d,0xc8,0xe0,0x4e]
// CHECK: fcmgt   v0.4h, v31.4h, #0.0     // encoding: [0xe0,0xcb,0xf8,0x0e]
// CHECK: fcmgt   v4.8h, v7.8h, #0.0      // encoding: [0xe4,0xc8,0xf8,0x4e]
// CHECK: fcmgt v0.2s, v31.2s, #0.0   // encoding: [0xe0,0xcb,0xa0,0x0e]
// CHECK: fcmgt v4.4s, v7.4s, #0.0    // encoding: [0xe4,0xc8,0xa0,0x4e]
// CHECK: fcmgt v29.2d, v2.2d, #0.0   // encoding: [0x5d,0xc8,0xe0,0x4e]

//----------------------------------------------------------------------
// Vector Compare Mask Less Than or Equal To Zero (Floating Point)
//----------------------------------------------------------------------
         fcmle v3.4h, v20.4h, #0.0
         fcmle v1.8h, v8.8h, #0.0
         fcmle v1.4s, v8.4s, #0.0
         fcmle v3.2s, v20.2s, #0.0
         fcmle v7.2d, v13.2d, #0.0
         fcmle v3.4h, v20.4h, #0
         fcmle v1.8h, v8.8h, #0
         fcmle v1.4s, v8.4s, #0
         fcmle v3.2s, v20.2s, #0
         fcmle v7.2d, v13.2d, #0

// CHECK: fcmle   v3.4h, v20.4h, #0.0     // encoding: [0x83,0xda,0xf8,0x2e]
// CHECK: fcmle   v1.8h, v8.8h, #0.0      // encoding: [0x01,0xd9,0xf8,0x6e]
// CHECK: fcmle v1.4s, v8.4s, #0.0   // encoding: [0x01,0xd9,0xa0,0x6e]
// CHECK: fcmle v3.2s, v20.2s, #0.0  // encoding: [0x83,0xda,0xa0,0x2e]
// CHECK: fcmle v7.2d, v13.2d, #0.0  // encoding: [0xa7,0xd9,0xe0,0x6e]
// CHECK: fcmle   v3.4h, v20.4h, #0.0     // encoding: [0x83,0xda,0xf8,0x2e]
// CHECK: fcmle   v1.8h, v8.8h, #0.0      // encoding: [0x01,0xd9,0xf8,0x6e]
// CHECK: fcmle v1.4s, v8.4s, #0.0   // encoding: [0x01,0xd9,0xa0,0x6e]
// CHECK: fcmle v3.2s, v20.2s, #0.0  // encoding: [0x83,0xda,0xa0,0x2e]
// CHECK: fcmle v7.2d, v13.2d, #0.0  // encoding: [0xa7,0xd9,0xe0,0x6e]

//----------------------------------------------------------------------
// Vector Compare Mask Less Than Zero (Floating Point)
//----------------------------------------------------------------------
         fcmlt v16.4h, v2.4h, #0.0
         fcmlt v15.8h, v4.8h, #0.0
         fcmlt v16.2s, v2.2s, #0.0
         fcmlt v15.4s, v4.4s, #0.0
         fcmlt v5.2d, v29.2d, #0.0
         fcmlt v16.4h, v2.4h, #0
         fcmlt v15.8h, v4.8h, #0
         fcmlt v16.2s, v2.2s, #0
         fcmlt v15.4s, v4.4s, #0
         fcmlt v5.2d, v29.2d, #0

// CHECK: fcmlt   v16.4h, v2.4h, #0.0     // encoding: [0x50,0xe8,0xf8,0x0e]
// CHECK: fcmlt   v15.8h, v4.8h, #0.0     // encoding: [0x8f,0xe8,0xf8,0x4e]
// CHECK: fcmlt v16.2s, v2.2s, #0.0   // encoding: [0x50,0xe8,0xa0,0x0e]
// CHECK: fcmlt v15.4s, v4.4s, #0.0   // encoding: [0x8f,0xe8,0xa0,0x4e]
// CHECK: fcmlt v5.2d, v29.2d, #0.0   // encoding: [0xa5,0xeb,0xe0,0x4e]
// CHECK: fcmlt   v16.4h, v2.4h, #0.0     // encoding: [0x50,0xe8,0xf8,0x0e]
// CHECK: fcmlt   v15.8h, v4.8h, #0.0     // encoding: [0x8f,0xe8,0xf8,0x4e]
// CHECK: fcmlt v16.2s, v2.2s, #0.0   // encoding: [0x50,0xe8,0xa0,0x0e]
// CHECK: fcmlt v15.4s, v4.4s, #0.0   // encoding: [0x8f,0xe8,0xa0,0x4e]
// CHECK: fcmlt v5.2d, v29.2d, #0.0   // encoding: [0xa5,0xeb,0xe0,0x4e]









