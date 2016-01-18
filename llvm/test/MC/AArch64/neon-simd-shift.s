// RUN: llvm-mc -triple=aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//------------------------------------------------------------------------------
// Vector shift right by immediate
//------------------------------------------------------------------------------
         sshr v0.8b, v1.8b, #3
         sshr v0.4h, v1.4h, #3
         sshr v0.2s, v1.2s, #3
         sshr v0.16b, v1.16b, #3
         sshr v0.8h, v1.8h, #3
         sshr v0.4s, v1.4s, #3
         sshr v0.2d, v1.2d, #3
// CHECK:	sshr	v0.8b, v1.8b, #3        // encoding: [0x20,0x04,0x0d,0x0f]
// CHECK:	sshr	v0.4h, v1.4h, #3        // encoding: [0x20,0x04,0x1d,0x0f]
// CHECK:	sshr	v0.2s, v1.2s, #3        // encoding: [0x20,0x04,0x3d,0x0f]
// CHECK:	sshr	v0.16b, v1.16b, #3      // encoding: [0x20,0x04,0x0d,0x4f]
// CHECK:	sshr	v0.8h, v1.8h, #3        // encoding: [0x20,0x04,0x1d,0x4f]
// CHECK:	sshr	v0.4s, v1.4s, #3        // encoding: [0x20,0x04,0x3d,0x4f]
// CHECK:	sshr	v0.2d, v1.2d, #3        // encoding: [0x20,0x04,0x7d,0x4f]

//------------------------------------------------------------------------------
// Vector  shift right by immediate
//------------------------------------------------------------------------------
         ushr v0.8b, v1.8b, #3
         ushr v0.4h, v1.4h, #3
         ushr v0.2s, v1.2s, #3
         ushr v0.16b, v1.16b, #3
         ushr v0.8h, v1.8h, #3
         ushr v0.4s, v1.4s, #3
         ushr v0.2d, v1.2d, #3

// CHECK: 	ushr	v0.8b, v1.8b, #3        // encoding: [0x20,0x04,0x0d,0x2f]
// CHECK: 	ushr	v0.4h, v1.4h, #3        // encoding: [0x20,0x04,0x1d,0x2f]
// CHECK:	ushr	v0.2s, v1.2s, #3        // encoding: [0x20,0x04,0x3d,0x2f]
// CHECK:	ushr	v0.16b, v1.16b, #3      // encoding: [0x20,0x04,0x0d,0x6f]
// CHECK:	ushr	v0.8h, v1.8h, #3        // encoding: [0x20,0x04,0x1d,0x6f]
// CHECK:	ushr	v0.4s, v1.4s, #3        // encoding: [0x20,0x04,0x3d,0x6f]
// CHECK:	ushr	v0.2d, v1.2d, #3        // encoding: [0x20,0x04,0x7d,0x6f]

//------------------------------------------------------------------------------
// Vector shift right and accumulate by immediate
//------------------------------------------------------------------------------
         ssra v0.8b, v1.8b, #3
         ssra v0.4h, v1.4h, #3
         ssra v0.2s, v1.2s, #3
         ssra v0.16b, v1.16b, #3
         ssra v0.8h, v1.8h, #3
         ssra v0.4s, v1.4s, #3
         ssra v0.2d, v1.2d, #3

// CHECK:	ssra	v0.8b, v1.8b, #3        // encoding: [0x20,0x14,0x0d,0x0f]
// CHECK:	ssra	v0.4h, v1.4h, #3        // encoding: [0x20,0x14,0x1d,0x0f]
// CHECK:	ssra	v0.2s, v1.2s, #3        // encoding: [0x20,0x14,0x3d,0x0f]
// CHECK:	ssra	v0.16b, v1.16b, #3      // encoding: [0x20,0x14,0x0d,0x4f]
// CHECK:	ssra	v0.8h, v1.8h, #3        // encoding: [0x20,0x14,0x1d,0x4f]
// CHECK:	ssra	v0.4s, v1.4s, #3        // encoding: [0x20,0x14,0x3d,0x4f]
// CHECK:	ssra	v0.2d, v1.2d, #3        // encoding: [0x20,0x14,0x7d,0x4f]

//------------------------------------------------------------------------------
// Vector  shift right and accumulate by immediate
//------------------------------------------------------------------------------
         usra v0.8b, v1.8b, #3
         usra v0.4h, v1.4h, #3
         usra v0.2s, v1.2s, #3
         usra v0.16b, v1.16b, #3
         usra v0.8h, v1.8h, #3
         usra v0.4s, v1.4s, #3
         usra v0.2d, v1.2d, #3

// CHECK:	usra	v0.8b, v1.8b, #3        // encoding: [0x20,0x14,0x0d,0x2f]
// CHECK:	usra	v0.4h, v1.4h, #3        // encoding: [0x20,0x14,0x1d,0x2f]
// CHECK:	usra	v0.2s, v1.2s, #3        // encoding: [0x20,0x14,0x3d,0x2f]
// CHECK:	usra	v0.16b, v1.16b, #3      // encoding: [0x20,0x14,0x0d,0x6f]
// CHECK:	usra	v0.8h, v1.8h, #3        // encoding: [0x20,0x14,0x1d,0x6f]
// CHECK:	usra	v0.4s, v1.4s, #3        // encoding: [0x20,0x14,0x3d,0x6f]
// CHECK:	usra	v0.2d, v1.2d, #3        // encoding: [0x20,0x14,0x7d,0x6f]

//------------------------------------------------------------------------------
// Vector rounding shift right by immediate
//------------------------------------------------------------------------------
         srshr v0.8b, v1.8b, #3
         srshr v0.4h, v1.4h, #3
         srshr v0.2s, v1.2s, #3
         srshr v0.16b, v1.16b, #3
         srshr v0.8h, v1.8h, #3
         srshr v0.4s, v1.4s, #3
         srshr v0.2d, v1.2d, #3

// CHECK:	srshr	v0.8b, v1.8b, #3        // encoding: [0x20,0x24,0x0d,0x0f]
// CHECK:	srshr	v0.4h, v1.4h, #3        // encoding: [0x20,0x24,0x1d,0x0f]
// CHECK:	srshr	v0.2s, v1.2s, #3        // encoding: [0x20,0x24,0x3d,0x0f]
// CHECK:	srshr	v0.16b, v1.16b, #3      // encoding: [0x20,0x24,0x0d,0x4f]
// CHECK:	srshr	v0.8h, v1.8h, #3        // encoding: [0x20,0x24,0x1d,0x4f]
// CHECK:	srshr	v0.4s, v1.4s, #3        // encoding: [0x20,0x24,0x3d,0x4f]
// CHECK:	srshr	v0.2d, v1.2d, #3        // encoding: [0x20,0x24,0x7d,0x4f]


//------------------------------------------------------------------------------
// Vecotr rounding shift right by immediate
//------------------------------------------------------------------------------
         urshr v0.8b, v1.8b, #3
         urshr v0.4h, v1.4h, #3
         urshr v0.2s, v1.2s, #3
         urshr v0.16b, v1.16b, #3
         urshr v0.8h, v1.8h, #3
         urshr v0.4s, v1.4s, #3
         urshr v0.2d, v1.2d, #3

// CHECK:	urshr	v0.8b, v1.8b, #3        // encoding: [0x20,0x24,0x0d,0x2f]
// CHECK:	urshr	v0.4h, v1.4h, #3        // encoding: [0x20,0x24,0x1d,0x2f]
// CHECK:	urshr	v0.2s, v1.2s, #3        // encoding: [0x20,0x24,0x3d,0x2f]
// CHECK:	urshr	v0.16b, v1.16b, #3      // encoding: [0x20,0x24,0x0d,0x6f]
// CHECK:	urshr	v0.8h, v1.8h, #3        // encoding: [0x20,0x24,0x1d,0x6f]
// CHECK:	urshr	v0.4s, v1.4s, #3        // encoding: [0x20,0x24,0x3d,0x6f]
// CHECK:	urshr	v0.2d, v1.2d, #3        // encoding: [0x20,0x24,0x7d,0x6f]


//------------------------------------------------------------------------------
// Vector rounding shift right and accumulate by immediate
//------------------------------------------------------------------------------
         srsra v0.8b, v1.8b, #3
         srsra v0.4h, v1.4h, #3
         srsra v0.2s, v1.2s, #3
         srsra v0.16b, v1.16b, #3
         srsra v0.8h, v1.8h, #3
         srsra v0.4s, v1.4s, #3
         srsra v0.2d, v1.2d, #3

// CHECK:	srsra	v0.8b, v1.8b, #3        // encoding: [0x20,0x34,0x0d,0x0f]
// CHECK:	srsra	v0.4h, v1.4h, #3        // encoding: [0x20,0x34,0x1d,0x0f]
// CHECK:	srsra	v0.2s, v1.2s, #3        // encoding: [0x20,0x34,0x3d,0x0f]
// CHECK:	srsra	v0.16b, v1.16b, #3      // encoding: [0x20,0x34,0x0d,0x4f]
// CHECK:	srsra	v0.8h, v1.8h, #3        // encoding: [0x20,0x34,0x1d,0x4f]
// CHECK:	srsra	v0.4s, v1.4s, #3        // encoding: [0x20,0x34,0x3d,0x4f]
// CHECK:	srsra	v0.2d, v1.2d, #3        // encoding: [0x20,0x34,0x7d,0x4f]


//------------------------------------------------------------------------------
// Vector rounding shift right and accumulate by immediate
//------------------------------------------------------------------------------
         ursra v0.8b, v1.8b, #3
         ursra v0.4h, v1.4h, #3
         ursra v0.2s, v1.2s, #3
         ursra v0.16b, v1.16b, #3
         ursra v0.8h, v1.8h, #3
         ursra v0.4s, v1.4s, #3
         ursra v0.2d, v1.2d, #3

// CHECK:	ursra	v0.8b, v1.8b, #3        // encoding: [0x20,0x34,0x0d,0x2f]
// CHECK:	ursra	v0.4h, v1.4h, #3        // encoding: [0x20,0x34,0x1d,0x2f]
// CHECK:	ursra	v0.2s, v1.2s, #3        // encoding: [0x20,0x34,0x3d,0x2f]
// CHECK:	ursra	v0.16b, v1.16b, #3      // encoding: [0x20,0x34,0x0d,0x6f]
// CHECK:	ursra	v0.8h, v1.8h, #3        // encoding: [0x20,0x34,0x1d,0x6f]
// CHECK:	ursra	v0.4s, v1.4s, #3        // encoding: [0x20,0x34,0x3d,0x6f]
// CHECK:	ursra	v0.2d, v1.2d, #3        // encoding: [0x20,0x34,0x7d,0x6f]


//------------------------------------------------------------------------------
// Vector shift right and insert by immediate
//------------------------------------------------------------------------------
         sri v0.8b, v1.8b, #3
         sri v0.4h, v1.4h, #3
         sri v0.2s, v1.2s, #3
         sri v0.16b, v1.16b, #3
         sri v0.8h, v1.8h, #3
         sri v0.4s, v1.4s, #3
         sri v0.2d, v1.2d, #3

// CHECK:	sri	v0.8b, v1.8b, #3        // encoding: [0x20,0x44,0x0d,0x2f]
// CHECK:	sri	v0.4h, v1.4h, #3        // encoding: [0x20,0x44,0x1d,0x2f]
// CHECK:	sri	v0.2s, v1.2s, #3        // encoding: [0x20,0x44,0x3d,0x2f]
// CHECK:	sri	v0.16b, v1.16b, #3      // encoding: [0x20,0x44,0x0d,0x6f]
// CHECK:	sri	v0.8h, v1.8h, #3        // encoding: [0x20,0x44,0x1d,0x6f]
// CHECK:	sri	v0.4s, v1.4s, #3        // encoding: [0x20,0x44,0x3d,0x6f]


//------------------------------------------------------------------------------
// Vector shift left and insert by immediate
//------------------------------------------------------------------------------
         sli v0.8b, v1.8b, #3
         sli v0.4h, v1.4h, #3
         sli v0.2s, v1.2s, #3
         sli v0.16b, v1.16b, #3
         sli v0.8h, v1.8h, #3
         sli v0.4s, v1.4s, #3
         sli v0.2d, v1.2d, #3

// CHECK:	sli	v0.8b, v1.8b, #3        // encoding: [0x20,0x54,0x0b,0x2f]
// CHECK:	sli	v0.4h, v1.4h, #3        // encoding: [0x20,0x54,0x13,0x2f]
// CHECK:	sli	v0.2s, v1.2s, #3        // encoding: [0x20,0x54,0x23,0x2f]
// CHECK:	sli	v0.16b, v1.16b, #3      // encoding: [0x20,0x54,0x0b,0x6f]
// CHECK:	sli	v0.8h, v1.8h, #3        // encoding: [0x20,0x54,0x13,0x6f]
// CHECK:	sli	v0.4s, v1.4s, #3        // encoding: [0x20,0x54,0x23,0x6f]
// CHECK:	sli	v0.2d, v1.2d, #3        // encoding: [0x20,0x54,0x43,0x6f]

//------------------------------------------------------------------------------
// Vector saturating shift left unsigned by immediate
//------------------------------------------------------------------------------
         sqshlu v0.8b, v1.8b, #3
         sqshlu v0.4h, v1.4h, #3
         sqshlu v0.2s, v1.2s, #3
         sqshlu v0.16b, v1.16b, #3
         sqshlu v0.8h, v1.8h, #3
         sqshlu v0.4s, v1.4s, #3
         sqshlu v0.2d, v1.2d, #3

// CHECK:	sqshlu	v0.8b, v1.8b, #3        // encoding: [0x20,0x64,0x0b,0x2f]
// CHECK:	sqshlu	v0.4h, v1.4h, #3        // encoding: [0x20,0x64,0x13,0x2f]
// CHECK:	sqshlu	v0.2s, v1.2s, #3        // encoding: [0x20,0x64,0x23,0x2f]
// CHECK:	sqshlu	v0.16b, v1.16b, #3      // encoding: [0x20,0x64,0x0b,0x6f]
// CHECK:	sqshlu	v0.8h, v1.8h, #3        // encoding: [0x20,0x64,0x13,0x6f]
// CHECK:	sqshlu	v0.4s, v1.4s, #3        // encoding: [0x20,0x64,0x23,0x6f]
// CHECK:	sqshlu	v0.2d, v1.2d, #3        // encoding: [0x20,0x64,0x43,0x6f]


//------------------------------------------------------------------------------
// Vector saturating shift left by immediate
//------------------------------------------------------------------------------
         sqshl v0.8b, v1.8b, #3
         sqshl v0.4h, v1.4h, #3
         sqshl v0.2s, v1.2s, #3
         sqshl v0.16b, v1.16b, #3
         sqshl v0.8h, v1.8h, #3
         sqshl v0.4s, v1.4s, #3
         sqshl v0.2d, v1.2d, #3

// CHECK:	sqshl	v0.8b, v1.8b, #3        // encoding: [0x20,0x74,0x0b,0x0f]
// CHECK:	sqshl	v0.4h, v1.4h, #3        // encoding: [0x20,0x74,0x13,0x0f]
// CHECK:	sqshl	v0.2s, v1.2s, #3        // encoding: [0x20,0x74,0x23,0x0f]
// CHECK:	sqshl	v0.16b, v1.16b, #3      // encoding: [0x20,0x74,0x0b,0x4f]
// CHECK:	sqshl	v0.8h, v1.8h, #3        // encoding: [0x20,0x74,0x13,0x4f]
// CHECK:	sqshl	v0.4s, v1.4s, #3        // encoding: [0x20,0x74,0x23,0x4f]
// CHECK:	sqshl	v0.2d, v1.2d, #3        // encoding: [0x20,0x74,0x43,0x4f]



//------------------------------------------------------------------------------
// Vector saturating shift left by immediate
//------------------------------------------------------------------------------
         uqshl v0.8b, v1.8b, #3
         uqshl v0.4h, v1.4h, #3
         uqshl v0.2s, v1.2s, #3
         uqshl v0.16b, v1.16b, #3
         uqshl v0.8h, v1.8h, #3
         uqshl v0.4s, v1.4s, #3
         uqshl v0.2d, v1.2d, #3

// CHECK:	uqshl	v0.8b, v1.8b, #3        // encoding: [0x20,0x74,0x0b,0x2f]
// CHECK:	uqshl	v0.4h, v1.4h, #3        // encoding: [0x20,0x74,0x13,0x2f]
// CHECK:	uqshl	v0.2s, v1.2s, #3        // encoding: [0x20,0x74,0x23,0x2f]
// CHECK:	uqshl	v0.16b, v1.16b, #3      // encoding: [0x20,0x74,0x0b,0x6f]
// CHECK:	uqshl	v0.8h, v1.8h, #3        // encoding: [0x20,0x74,0x13,0x6f]
// CHECK:	uqshl	v0.4s, v1.4s, #3        // encoding: [0x20,0x74,0x23,0x6f]
// CHECK:	uqshl	v0.2d, v1.2d, #3        // encoding: [0x20,0x74,0x43,0x6f]


//------------------------------------------------------------------------------
// Vector shift right narrow by immediate
//------------------------------------------------------------------------------
         shrn v0.8b, v1.8h, #3
         shrn v0.4h, v1.4s, #3
         shrn v0.2s, v1.2d, #3
         shrn2 v0.16b, v1.8h, #3
         shrn2 v0.8h, v1.4s, #3
         shrn2 v0.4s, v1.2d, #3

// CHECK:	shrn	v0.8b, v1.8h, #3        // encoding: [0x20,0x84,0x0d,0x0f]
// CHECK:	shrn	v0.4h, v1.4s, #3        // encoding: [0x20,0x84,0x1d,0x0f]
// CHECK:	shrn	v0.2s, v1.2d, #3        // encoding: [0x20,0x84,0x3d,0x0f]
// CHECK:	shrn2	v0.16b, v1.8h, #3       // encoding: [0x20,0x84,0x0d,0x4f]
// CHECK:	shrn2	v0.8h, v1.4s, #3        // encoding: [0x20,0x84,0x1d,0x4f]
// CHECK:	shrn2	v0.4s, v1.2d, #3        // encoding: [0x20,0x84,0x3d,0x4f]

//------------------------------------------------------------------------------
// Vector saturating shift right unsigned narrow by immediate
//------------------------------------------------------------------------------
         sqshrun v0.8b, v1.8h, #3
         sqshrun v0.4h, v1.4s, #3
         sqshrun v0.2s, v1.2d, #3
         sqshrun2 v0.16b, v1.8h, #3
         sqshrun2 v0.8h, v1.4s, #3
         sqshrun2 v0.4s, v1.2d, #3

// CHECK:	sqshrun	v0.8b, v1.8h, #3        // encoding: [0x20,0x84,0x0d,0x2f]
// CHECK:	sqshrun	v0.4h, v1.4s, #3        // encoding: [0x20,0x84,0x1d,0x2f]
// CHECK:	sqshrun	v0.2s, v1.2d, #3        // encoding: [0x20,0x84,0x3d,0x2f]
// CHECK:	sqshrun2	v0.16b, v1.8h, #3 	// encoding: [0x20,0x84,0x0d,0x6f]
// CHECK:	sqshrun2	v0.8h, v1.4s, #3 	// encoding: [0x20,0x84,0x1d,0x6f]
// CHECK:	sqshrun2	v0.4s, v1.2d, #3 	// encoding: [0x20,0x84,0x3d,0x6f]

//------------------------------------------------------------------------------
// Vector rounding shift right narrow by immediate
//------------------------------------------------------------------------------
         rshrn v0.8b, v1.8h, #3
         rshrn v0.4h, v1.4s, #3
         rshrn v0.2s, v1.2d, #3
         rshrn2 v0.16b, v1.8h, #3
         rshrn2 v0.8h, v1.4s, #3
         rshrn2 v0.4s, v1.2d, #3

// CHECK:	rshrn	v0.8b, v1.8h, #3        // encoding: [0x20,0x8c,0x0d,0x0f]
// CHECK:	rshrn	v0.4h, v1.4s, #3        // encoding: [0x20,0x8c,0x1d,0x0f]
// CHECK:	rshrn	v0.2s, v1.2d, #3        // encoding: [0x20,0x8c,0x3d,0x0f]
// CHECK:	rshrn2	v0.16b, v1.8h, #3       // encoding: [0x20,0x8c,0x0d,0x4f]
// CHECK:	rshrn2	v0.8h, v1.4s, #3        // encoding: [0x20,0x8c,0x1d,0x4f]
// CHECK:	rshrn2	v0.4s, v1.2d, #3        // encoding: [0x20,0x8c,0x3d,0x4f]


//------------------------------------------------------------------------------
// Vector saturating shift right rounded unsigned narrow by immediate
//------------------------------------------------------------------------------
         sqrshrun v0.8b, v1.8h, #3
         sqrshrun v0.4h, v1.4s, #3
         sqrshrun v0.2s, v1.2d, #3
         sqrshrun2 v0.16b, v1.8h, #3
         sqrshrun2 v0.8h, v1.4s, #3
         sqrshrun2 v0.4s, v1.2d, #3

// CHECK:	sqrshrun	v0.8b, v1.8h, #3    // encoding: [0x20,0x8c,0x0d,0x2f]
// CHECK:	sqrshrun	v0.4h, v1.4s, #3    // encoding: [0x20,0x8c,0x1d,0x2f]
// CHECK:	sqrshrun	v0.2s, v1.2d, #3    // encoding: [0x20,0x8c,0x3d,0x2f]
// CHECK:	sqrshrun2	v0.16b, v1.8h, #3   // encoding: [0x20,0x8c,0x0d,0x6f]
// CHECK:	sqrshrun2	v0.8h, v1.4s, #3    // encoding: [0x20,0x8c,0x1d,0x6f]
// CHECK:	sqrshrun2	v0.4s, v1.2d, #3    // encoding: [0x20,0x8c,0x3d,0x6f]


//------------------------------------------------------------------------------
// Vector saturating shift right narrow by immediate
//------------------------------------------------------------------------------
         sqshrn v0.8b, v1.8h, #3
         sqshrn v0.4h, v1.4s, #3
         sqshrn v0.2s, v1.2d, #3
         sqshrn2 v0.16b, v1.8h, #3
         sqshrn2 v0.8h, v1.4s, #3
         sqshrn2 v0.4s, v1.2d, #3

// CHECK:	sqshrn	v0.8b, v1.8h, #3        // encoding: [0x20,0x94,0x0d,0x0f]
// CHECK:	sqshrn	v0.4h, v1.4s, #3        // encoding: [0x20,0x94,0x1d,0x0f]
// CHECK:	sqshrn	v0.2s, v1.2d, #3        // encoding: [0x20,0x94,0x3d,0x0f]
// CHECK:	sqshrn2	v0.16b, v1.8h, #3       // encoding: [0x20,0x94,0x0d,0x4f]
// CHECK:	sqshrn2	v0.8h, v1.4s, #3        // encoding: [0x20,0x94,0x1d,0x4f]
// CHECK:	sqshrn2	v0.4s, v1.2d, #3        // encoding: [0x20,0x94,0x3d,0x4f]


//------------------------------------------------------------------------------
// Vector saturating shift right narrow by immediate
//------------------------------------------------------------------------------
         uqshrn v0.8b, v1.8h, #3
         uqshrn v0.4h, v1.4s, #3
         uqshrn v0.2s, v1.2d, #3
         uqshrn2 v0.16b, v1.8h, #3
         uqshrn2 v0.8h, v1.4s, #3
         uqshrn2 v0.4s, v1.2d, #3

// CHECK:	uqshrn	v0.8b, v1.8h, #3        // encoding: [0x20,0x94,0x0d,0x2f]
// CHECK:	uqshrn	v0.4h, v1.4s, #3        // encoding: [0x20,0x94,0x1d,0x2f]
// CHECK:	uqshrn	v0.2s, v1.2d, #3        // encoding: [0x20,0x94,0x3d,0x2f]
// CHECK:	uqshrn2	v0.16b, v1.8h, #3       // encoding: [0x20,0x94,0x0d,0x6f]
// CHECK:	uqshrn2	v0.8h, v1.4s, #3        // encoding: [0x20,0x94,0x1d,0x6f]
// CHECK:	uqshrn2	v0.4s, v1.2d, #3        // encoding: [0x20,0x94,0x3d,0x6f]

//------------------------------------------------------------------------------
// Vector saturating shift right rounded narrow by immediate
//------------------------------------------------------------------------------
         sqrshrn v0.8b, v1.8h, #3
         sqrshrn v0.4h, v1.4s, #3
         sqrshrn v0.2s, v1.2d, #3
         sqrshrn2 v0.16b, v1.8h, #3
         sqrshrn2 v0.8h, v1.4s, #3
         sqrshrn2 v0.4s, v1.2d, #3

// CHECK:	sqrshrn	v0.8b, v1.8h, #3        // encoding: [0x20,0x9c,0x0d,0x0f]
// CHECK:	sqrshrn	v0.4h, v1.4s, #3        // encoding: [0x20,0x9c,0x1d,0x0f]
// CHECK:	sqrshrn	v0.2s, v1.2d, #3        // encoding: [0x20,0x9c,0x3d,0x0f]
// CHECK:	sqrshrn2	v0.16b, v1.8h, #3   // encoding: [0x20,0x9c,0x0d,0x4f]
// CHECK:	sqrshrn2	v0.8h, v1.4s, #3    // encoding: [0x20,0x9c,0x1d,0x4f]
// CHECK:	sqrshrn2	v0.4s, v1.2d, #3    // encoding: [0x20,0x9c,0x3d,0x4f]


//------------------------------------------------------------------------------
// Vector saturating shift right rounded narrow by immediate
//------------------------------------------------------------------------------
         uqrshrn v0.8b, v1.8h, #3
         uqrshrn v0.4h, v1.4s, #3
         uqrshrn v0.2s, v1.2d, #3
         uqrshrn2 v0.16b, v1.8h, #3
         uqrshrn2 v0.8h, v1.4s, #3
         uqrshrn2 v0.4s, v1.2d, #3

// CHECK:	uqrshrn	v0.8b, v1.8h, #3        // encoding: [0x20,0x9c,0x0d,0x2f]
// CHECK:	uqrshrn	v0.4h, v1.4s, #3        // encoding: [0x20,0x9c,0x1d,0x2f]
// CHECK:	uqrshrn	v0.2s, v1.2d, #3        // encoding: [0x20,0x9c,0x3d,0x2f]
// CHECK:	uqrshrn2	v0.16b, v1.8h, #3   // encoding: [0x20,0x9c,0x0d,0x6f]
// CHECK:	uqrshrn2	v0.8h, v1.4s, #3    // encoding: [0x20,0x9c,0x1d,0x6f]
// CHECK:	uqrshrn2	v0.4s, v1.2d, #3    // encoding: [0x20,0x9c,0x3d,0x6f]


//------------------------------------------------------------------------------
// Fixed-point convert to floating-point
//------------------------------------------------------------------------------
         scvtf v0.2s, v1.2s, #3
         scvtf v0.4s, v1.4s, #3
         scvtf v0.2d, v1.2d, #3
         ucvtf v0.2s, v1.2s, #3
         ucvtf v0.4s, v1.4s, #3
         ucvtf v0.2d, v1.2d, #3

// CHECK:	scvtf	v0.2s, v1.2s, #3        // encoding: [0x20,0xe4,0x3d,0x0f]
// CHECK:	scvtf	v0.4s, v1.4s, #3        // encoding: [0x20,0xe4,0x3d,0x4f]
// CHECK:	scvtf	v0.2d, v1.2d, #3        // encoding: [0x20,0xe4,0x7d,0x4f]
// CHECK:	ucvtf	v0.2s, v1.2s, #3        // encoding: [0x20,0xe4,0x3d,0x2f]
// CHECK:	ucvtf	v0.4s, v1.4s, #3        // encoding: [0x20,0xe4,0x3d,0x6f]
// CHECK:	ucvtf	v0.2d, v1.2d, #3        // encoding: [0x20,0xe4,0x7d,0x6f]

//------------------------------------------------------------------------------
// Floating-point convert to fixed-point
//------------------------------------------------------------------------------
         fcvtzs v0.2s, v1.2s, #3
         fcvtzs v0.4s, v1.4s, #3
         fcvtzs v0.2d, v1.2d, #3
         fcvtzu v0.2s, v1.2s, #3
         fcvtzu v0.4s, v1.4s, #3
         fcvtzu v0.2d, v1.2d, #3


// CHECK:	fcvtzs	v0.2s, v1.2s, #3        // encoding: [0x20,0xfc,0x3d,0x0f]
// CHECK:	fcvtzs	v0.4s, v1.4s, #3        // encoding: [0x20,0xfc,0x3d,0x4f]
// CHECK:	fcvtzs	v0.2d, v1.2d, #3        // encoding: [0x20,0xfc,0x7d,0x4f]
// CHECK:	fcvtzu	v0.2s, v1.2s, #3        // encoding: [0x20,0xfc,0x3d,0x2f]
// CHECK:	fcvtzu	v0.4s, v1.4s, #3        // encoding: [0x20,0xfc,0x3d,0x6f]
// CHECK:	fcvtzu	v0.2d, v1.2d, #3        // encoding: [0x20,0xfc,0x7d,0x6f]

