// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//----------------------------------------------------------------------
// Scalar Signed Shift Right (Immediate)
//----------------------------------------------------------------------
        sshr d15, d16, #12

// CHECK: sshr d15, d16, #12  // encoding: [0x0f,0x06,0x74,0x5f]

//----------------------------------------------------------------------
// Scalar Unsigned Shift Right (Immediate)
//----------------------------------------------------------------------
        ushr d10, d17, #18

// CHECK: ushr d10, d17, #18  // encoding: [0x2a,0x06,0x6e,0x7f]

//----------------------------------------------------------------------
// Scalar Signed Rounding Shift Right (Immediate)
//----------------------------------------------------------------------
        srshr d19, d18, #7

// CHECK: srshr d19, d18, #7  // encoding: [0x53,0x26,0x79,0x5f]

//----------------------------------------------------------------------
// Scalar Unigned Rounding Shift Right (Immediate)
//----------------------------------------------------------------------
        urshr d20, d23, #31

// CHECK: urshr d20, d23, #31  // encoding: [0xf4,0x26,0x61,0x7f]

//----------------------------------------------------------------------
// Scalar Signed Shift Right and Accumulate (Immediate)
//----------------------------------------------------------------------
        ssra d18, d12, #21

// CHECK: ssra d18, d12, #21  // encoding: [0x92,0x15,0x6b,0x5f]

//----------------------------------------------------------------------
// Scalar Unsigned Shift Right and Accumulate (Immediate)
//----------------------------------------------------------------------
        usra d20, d13, #61

// CHECK: usra d20, d13, #61  // encoding: [0xb4,0x15,0x43,0x7f]

//----------------------------------------------------------------------
// Scalar Signed Rounding Shift Right and Accumulate (Immediate)
//----------------------------------------------------------------------
        srsra d15, d11, #19

// CHECK: srsra d15, d11, #19  // encoding: [0x6f,0x35,0x6d,0x5f]

//----------------------------------------------------------------------
// Scalar Unsigned Rounding Shift Right and Accumulate (Immediate)
//----------------------------------------------------------------------
        ursra d18, d10, #13

// CHECK: ursra d18, d10, #13  // encoding: [0x52,0x35,0x73,0x7f]

//----------------------------------------------------------------------
// Scalar Shift Left (Immediate)
//----------------------------------------------------------------------
        shl d7, d10, #12

// CHECK: shl d7, d10, #12  // encoding: [0x47,0x55,0x4c,0x5f]

//----------------------------------------------------------------------
// Signed Saturating Shift Left (Immediate)
//----------------------------------------------------------------------
        sqshl b11, b19, #7
        sqshl h13, h18, #11
        sqshl s14, s17, #22
        sqshl d15, d16, #51

// CHECK: sqshl b11, b19, #7   // encoding: [0x6b,0x76,0x0f,0x5f]
// CHECK: sqshl h13, h18, #11  // encoding: [0x4d,0x76,0x1b,0x5f]
// CHECK: sqshl s14, s17, #22  // encoding: [0x2e,0x76,0x36,0x5f]
// CHECK: sqshl d15, d16, #51  // encoding: [0x0f,0x76,0x73,0x5f]

//----------------------------------------------------------------------
// Unsigned Saturating Shift Left (Immediate)
//----------------------------------------------------------------------
        uqshl b18, b15, #6
        uqshl h11, h18, #7
        uqshl s14, s19, #18
        uqshl d15, d12, #19

// CHECK: uqshl b18, b15, #6   // encoding: [0xf2,0x75,0x0e,0x7f]
// CHECK: uqshl h11, h18, #7   // encoding: [0x4b,0x76,0x17,0x7f]
// CHECK: uqshl s14, s19, #18  // encoding: [0x6e,0x76,0x32,0x7f]
// CHECK: uqshl d15, d12, #19  // encoding: [0x8f,0x75,0x53,0x7f]

//----------------------------------------------------------------------
// Signed Saturating Shift Left Unsigned (Immediate)
//----------------------------------------------------------------------
        sqshlu b15, b18, #6
        sqshlu h19, h17, #6
        sqshlu s16, s14, #25
        sqshlu d11, d13, #32

// CHECK: sqshlu  b15, b18, #6   // encoding: [0x4f,0x66,0x0e,0x7f]
// CHECK: sqshlu  h19, h17, #6   // encoding: [0x33,0x66,0x16,0x7f]
// CHECK: sqshlu  s16, s14, #25  // encoding: [0xd0,0x65,0x39,0x7f]
// CHECK: sqshlu  d11, d13, #32  // encoding: [0xab,0x65,0x60,0x7f]

//----------------------------------------------------------------------
// Shift Right And Insert (Immediate)
//----------------------------------------------------------------------
        sri d10, d12, #14

// CHECK: sri d10, d12, #14  // encoding: [0x8a,0x45,0x72,0x7f]

//----------------------------------------------------------------------
// Shift Left And Insert (Immediate)
//----------------------------------------------------------------------
        sli d10, d14, #12

// CHECK: sli d10, d14, #12  // encoding: [0xca,0x55,0x4c,0x7f]

//----------------------------------------------------------------------
// Signed Saturating Shift Right Narrow (Immediate)
//----------------------------------------------------------------------
        sqshrn b10, h15, #5
        sqshrn h17, s10, #4
        sqshrn s18, d10, #31

// CHECK: sqshrn  b10, h15, #5   // encoding: [0xea,0x95,0x0b,0x5f]
// CHECK: sqshrn  h17, s10, #4   // encoding: [0x51,0x95,0x1c,0x5f]
// CHECK: sqshrn  s18, d10, #31  // encoding: [0x52,0x95,0x21,0x5f]

//----------------------------------------------------------------------
// Unsigned Saturating Shift Right Narrow (Immediate)
//----------------------------------------------------------------------
        uqshrn b12, h10, #7
        uqshrn h10, s14, #5
        uqshrn s10, d12, #13

// CHECK: uqshrn  b12, h10, #7   // encoding: [0x4c,0x95,0x09,0x7f]
// CHECK: uqshrn  h10, s14, #5   // encoding: [0xca,0x95,0x1b,0x7f]
// CHECK: uqshrn  s10, d12, #13  // encoding: [0x8a,0x95,0x33,0x7f]

//----------------------------------------------------------------------
// Signed Saturating Rounded Shift Right Narrow (Immediate)
//----------------------------------------------------------------------
        sqrshrn b10, h13, #2
        sqrshrn h15, s10, #6
        sqrshrn s15, d12, #9

// CHECK: sqrshrn b10, h13, #2  // encoding: [0xaa,0x9d,0x0e,0x5f]
// CHECK: sqrshrn h15, s10, #6  // encoding: [0x4f,0x9d,0x1a,0x5f]
// CHECK: sqrshrn s15, d12, #9  // encoding: [0x8f,0x9d,0x37,0x5f]

//----------------------------------------------------------------------
// Unsigned Saturating Rounded Shift Right Narrow (Immediate)
//----------------------------------------------------------------------
        uqrshrn b10, h12, #5
        uqrshrn h12, s10, #14
        uqrshrn s10, d10, #25

// CHECK: uqrshrn b10, h12, #5   // encoding: [0x8a,0x9d,0x0b,0x7f]
// CHECK: uqrshrn h12, s10, #14  // encoding: [0x4c,0x9d,0x12,0x7f]
// CHECK: uqrshrn s10, d10, #25  // encoding: [0x4a,0x9d,0x27,0x7f]

//----------------------------------------------------------------------
// Signed Saturating Shift Right Unsigned Narrow (Immediate)
//----------------------------------------------------------------------
        sqshrun b15, h10, #7
        sqshrun h20, s14, #3
        sqshrun s10, d15, #15

// CHECK: sqshrun b15, h10, #7   // encoding: [0x4f,0x85,0x09,0x7f]
// CHECK: sqshrun h20, s14, #3   // encoding: [0xd4,0x85,0x1d,0x7f]
// CHECK: sqshrun s10, d15, #15  // encoding: [0xea,0x85,0x31,0x7f]

//----------------------------------------------------------------------
// Signed Saturating Rounded Shift Right Unsigned Narrow (Immediate)
//----------------------------------------------------------------------

        sqrshrun b17, h10, #6
        sqrshrun h10, s13, #15
        sqrshrun s22, d16, #31

// CHECK: sqrshrun b17, h10, #6   // encoding: [0x51,0x8d,0x0a,0x7f]
// CHECK: sqrshrun h10, s13, #15  // encoding: [0xaa,0x8d,0x11,0x7f]
// CHECK: sqrshrun s22, d16, #31  // encoding: [0x16,0x8e,0x21,0x7f]
