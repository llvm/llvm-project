// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//----------------------------------------------------------------------
// Scalar Floating-point Compare Mask Equal
//----------------------------------------------------------------------

         fcmeq s10, s11, s12
         fcmeq d20, d21, d22

// CHECK: fcmeq s10, s11, s12   // encoding: [0x6a,0xe5,0x2c,0x5e]
// CHECK: fcmeq d20, d21, d22   // encoding: [0xb4,0xe6,0x76,0x5e]

//----------------------------------------------------------------------
// Scalar Floating-point Compare Mask Equal To Zero
//----------------------------------------------------------------------

         fcmeq s10, s11, #0.0
         fcmeq d20, d21, #0.0
         fcmeq s10, s11, #0
         fcmeq d20, d21, #0x0

// CHECK: fcmeq s10, s11, #0.0   // encoding: [0x6a,0xd9,0xa0,0x5e]
// CHECK: fcmeq d20, d21, #0.0   // encoding: [0xb4,0xda,0xe0,0x5e]
// CHECK: fcmeq s10, s11, #0.0   // encoding: [0x6a,0xd9,0xa0,0x5e]
// CHECK: fcmeq d20, d21, #0.0   // encoding: [0xb4,0xda,0xe0,0x5e]

//----------------------------------------------------------------------
// Scalar Floating-point Compare Mask Greater Than Or Equal
//----------------------------------------------------------------------

         fcmge s10, s11, s12
         fcmge d20, d21, d22

// CHECK: fcmge s10, s11, s12   // encoding: [0x6a,0xe5,0x2c,0x7e]
// CHECK: fcmge d20, d21, d22   // encoding: [0xb4,0xe6,0x76,0x7e]

//----------------------------------------------------------------------
// Scalar Floating-point Compare Mask Greater Than Or Equal To Zero
//----------------------------------------------------------------------

         fcmge s10, s11, #0.0
         fcmge d20, d21, #0.0
         fcmge s10, s11, #0
         fcmge d20, d21, #0x0

// CHECK: fcmge s10, s11, #0.0   // encoding: [0x6a,0xc9,0xa0,0x7e]
// CHECK: fcmge d20, d21, #0.0   // encoding: [0xb4,0xca,0xe0,0x7e]
// CHECK: fcmge s10, s11, #0.0   // encoding: [0x6a,0xc9,0xa0,0x7e]
// CHECK: fcmge d20, d21, #0.0   // encoding: [0xb4,0xca,0xe0,0x7e]

//----------------------------------------------------------------------
// Scalar Floating-point Compare Mask Greather Than
//----------------------------------------------------------------------

         fcmgt s10, s11, s12
         fcmgt d20, d21, d22

// CHECK: fcmgt s10, s11, s12   // encoding: [0x6a,0xe5,0xac,0x7e]
// CHECK: fcmgt d20, d21, d22   // encoding: [0xb4,0xe6,0xf6,0x7e]

//----------------------------------------------------------------------
// Scalar Floating-point Compare Mask Greather Than Zero
//----------------------------------------------------------------------

         fcmgt s10, s11, #0.0
         fcmgt d20, d21, #0.0
         fcmgt s10, s11, #0
         fcmgt d20, d21, #0x0

// CHECK: fcmgt s10, s11, #0.0   // encoding: [0x6a,0xc9,0xa0,0x5e]
// CHECK: fcmgt d20, d21, #0.0   // encoding: [0xb4,0xca,0xe0,0x5e]
// CHECK: fcmgt s10, s11, #0.0   // encoding: [0x6a,0xc9,0xa0,0x5e]
// CHECK: fcmgt d20, d21, #0.0   // encoding: [0xb4,0xca,0xe0,0x5e]

//----------------------------------------------------------------------
// Scalar Floating-point Compare Mask Less Than Or Equal To Zero
//----------------------------------------------------------------------

         fcmle s10, s11, #0.0
         fcmle d20, d21, #0.0
         fcmle s10, s11, #0
         fcmle d20, d21, #0x0

// CHECK: fcmle s10, s11, #0.0   // encoding: [0x6a,0xd9,0xa0,0x7e]
// CHECK: fcmle d20, d21, #0.0   // encoding: [0xb4,0xda,0xe0,0x7e]
// CHECK: fcmle s10, s11, #0.0   // encoding: [0x6a,0xd9,0xa0,0x7e]
// CHECK: fcmle d20, d21, #0.0   // encoding: [0xb4,0xda,0xe0,0x7e]

//----------------------------------------------------------------------
// Scalar Floating-point Compare Mask Less Than
//----------------------------------------------------------------------

         fcmlt s10, s11, #0.0
         fcmlt d20, d21, #0.0
         fcmlt s10, s11, #0
         fcmlt d20, d21, #0x0

// CHECK: fcmlt s10, s11, #0.0   // encoding: [0x6a,0xe9,0xa0,0x5e]
// CHECK: fcmlt d20, d21, #0.0   // encoding: [0xb4,0xea,0xe0,0x5e]
// CHECK: fcmlt s10, s11, #0.0   // encoding: [0x6a,0xe9,0xa0,0x5e]
// CHECK: fcmlt d20, d21, #0.0   // encoding: [0xb4,0xea,0xe0,0x5e]

//----------------------------------------------------------------------
// Scalar Floating-point Absolute Compare Mask Greater Than Or Equal
//----------------------------------------------------------------------

         facge s10, s11, s12
         facge d20, d21, d22

// CHECK: facge s10, s11, s12    // encoding: [0x6a,0xed,0x2c,0x7e]
// CHECK: facge d20, d21, d22    // encoding: [0xb4,0xee,0x76,0x7e]

//----------------------------------------------------------------------
// Scalar Floating-point Absolute Compare Mask Greater Than
//----------------------------------------------------------------------

         facgt s10, s11, s12
         facgt d20, d21, d22

// CHECK: facgt s10, s11, s12   // encoding: [0x6a,0xed,0xac,0x7e]
// CHECK: facgt d20, d21, d22   // encoding: [0xb4,0xee,0xf6,0x7e]
