// RUN: not llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ERROR < %t %s

//------------------------------------------------------------------------------
// Vector Integer Add/sub
//------------------------------------------------------------------------------

        // Mismatched vector types
        add v0.16b, v1.8b, v2.8b
        sub v0.2d, v1.2d, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         add v0.16b, v1.8b, v2.8b
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sub v0.2d, v1.2d, v2.2s
// CHECK-ERROR:                              ^

//------------------------------------------------------------------------------
// Vector Floating-Point Add/sub
//------------------------------------------------------------------------------

        // Mismatched and invalid vector types
        fadd v0.2d, v1.2s, v2.2s
        fsub v0.4s, v1.2s, v2.4s
        fsub v0.8b, v1.8b, v2.8b

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fadd v0.2d, v1.2s, v2.2s
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fsub v0.4s, v1.2s, v2.4s
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fsub v0.8b, v1.8b, v2.8b
// CHECK-ERROR:                  ^

//----------------------------------------------------------------------
// Vector Integer Mul
//----------------------------------------------------------------------

        // Mismatched and invalid vector types
        mul v0.16b, v1.8b, v2.8b
        mul v0.2d, v1.2d, v2.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         mul v0.16b, v1.8b, v2.8b
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         mul v0.2d, v1.2d, v2.2d
// CHECK-ERROR:                ^

//----------------------------------------------------------------------
// Vector Floating-Point Mul/Div
//----------------------------------------------------------------------
        // Mismatched vector types
        fmul v0.16b, v1.8b, v2.8b
        fdiv v0.2s, v1.2d, v2.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fmul v0.16b, v1.8b, v2.8b
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fdiv v0.2s, v1.2d, v2.2d
// CHECK-ERROR:                        ^

//----------------------------------------------------------------------
// Vector And Orr Eor Bsl Bit Bif, Orn, Bic,
//----------------------------------------------------------------------
        // Mismatched and invalid vector types
        and v0.8b, v1.16b, v2.8b
        orr v0.4h, v1.4h, v2.4h
        eor v0.2s, v1.2s, v2.2s
        bsl v0.8b, v1.16b, v2.8b
        bsl v0.2s, v1.2s, v2.2s
        bit v0.2d, v1.2d, v2.2d
        bif v0.4h, v1.4h, v2.4h
        orn v0.8b, v1.16b, v2.16b
        bic v0.2d, v1.2d, v2.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         and v0.8b, v1.16b, v2.8b
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: immediate must be an integer in range [0, 255]
// CHECK-ERROR:         orr v0.4h, v1.4h, v2.4h
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         eor v0.2s, v1.2s, v2.2s
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         bsl v0.8b, v1.16b, v2.8b
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         bsl v0.2s, v1.2s, v2.2s
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         bit v0.2d, v1.2d, v2.2d
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         bif v0.4h, v1.4h, v2.4h
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         orn v0.8b, v1.16b, v2.16b
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         bic v0.2d, v1.2d, v2.2d
// CHECK-ERROR:                ^

//----------------------------------------------------------------------
// Vector Integer Multiply-accumulate and Multiply-subtract
//----------------------------------------------------------------------

        // Mismatched and invalid vector types
        mla v0.16b, v1.8b, v2.8b
        mls v0.2d, v1.2d, v2.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         mla v0.16b, v1.8b, v2.8b
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         mls v0.2d, v1.2d, v2.2d
// CHECK-ERROR:                ^

//----------------------------------------------------------------------
// Vector Floating-Point Multiply-accumulate and Multiply-subtract
//----------------------------------------------------------------------
        // Mismatched vector types
        fmla v0.2s, v1.2d, v2.2d
        fmls v0.16b, v1.8b, v2.8b

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fmla v0.2s, v1.2d, v2.2d
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fmls v0.16b, v1.8b, v2.8b
// CHECK-ERROR:                         ^


//----------------------------------------------------------------------
// Vector Move Immediate Shifted
// Vector Move Inverted Immediate Shifted
// Vector Bitwise Bit Clear (AND NOT) - immediate
// Vector Bitwise OR - immedidate
//----------------------------------------------------------------------
      // out of range immediate (0 to 0xff)
      movi v0.2s, #-1
      mvni v1.4s, #256
      // out of range shift (0, 8, 16, 24 and 0, 8)
      bic v15.4h, #1, lsl #7
      orr v31.2s, #1, lsl #25
      movi v5.4h, #10, lsl #16
      // invalid vector type (2s, 4s, 4h, 8h)
      movi v5.8b, #1, lsl #8

// CHECK-ERROR: error: immediate must be an integer in range [0, 255]
// CHECK-ERROR:          movi v0.2s, #-1
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: immediate must be an integer in range [0, 255]
// CHECK-ERROR:         mvni v1.4s, #256
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         bic v15.4h, #1, lsl #7
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         orr v31.2s, #1, lsl #25
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         movi v5.4h, #10, lsl #16
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         movi v5.8b, #1, lsl #8
// CHECK-ERROR:                         ^
//----------------------------------------------------------------------
// Vector Move Immediate Masked
// Vector Move Inverted Immediate Masked
//----------------------------------------------------------------------
      // out of range immediate (0 to 0xff)
      movi v0.2s, #-1, msl #8
      mvni v7.4s, #256, msl #16
      // out of range shift (8, 16)
      movi v3.2s, #1, msl #0
      mvni v17.4s, #255, msl #32
      // invalid vector type (2s, 4s)
      movi v5.4h, #31, msl #8

// CHECK-ERROR: error: immediate must be an integer in range [0, 255]
// CHECK-ERROR:         movi v0.2s, #-1, msl #8
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: immediate must be an integer in range [0, 255]
// CHECK-ERROR:         mvni v7.4s, #256, msl #16
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         movi v3.2s, #1, msl #0
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         mvni v17.4s, #255, msl #32
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         movi v5.4h, #31, msl #8
// CHECK-ERROR:                          ^

//----------------------------------------------------------------------
// Vector Immediate - per byte
//----------------------------------------------------------------------
        // out of range immediate (0 to 0xff)
        movi v0.8b, #-1
        movi v1.16b, #256

// CHECK-ERROR: error: immediate must be an integer in range [0, 255]
// CHECK-ERROR:         movi v0.8b, #-1
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: immediate must be an integer in range [0, 255]
// CHECK-ERROR:         movi v1.16b, #256
// CHECK-ERROR:                      ^

//----------------------------------------------------------------------
// Scalar Floating-point Reciprocal Estimate
//----------------------------------------------------------------------

    frecpe s19, h14
    frecpe d13, s13

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        frecpe s19, h14
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        frecpe d13, s13
// CHECK-ERROR:                    ^

//----------------------------------------------------------------------
// Scalar Floating-point Reciprocal Exponent
//----------------------------------------------------------------------

    frecpx s18, h10
    frecpx d16, s19

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        frecpx s18, h10
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        frecpx d16, s19
// CHECK-ERROR:                    ^

//----------------------------------------------------------------------
// Scalar Floating-point Reciprocal Square Root Estimate
//----------------------------------------------------------------------

    frsqrte s22, h13
    frsqrte d21, s12

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        frsqrte s22, h13
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        frsqrte d21, s12
// CHECK-ERROR:                     ^

//----------------------------------------------------------------------
// Vector Move Immediate - bytemask, per doubleword
//---------------------------------------------------------------------
        // invalid bytemask (0x00 or 0xff)
        movi v0.2d, #0x10ff00ff00ff00ff

// CHECK:ERROR: error: invalid operand for instruction
// CHECK:ERROR:         movi v0.2d, #0x10ff00ff00ff00ff
// CHECK:ERROR:                     ^

//----------------------------------------------------------------------
// Vector Move Immediate - bytemask, one doubleword
//----------------------------------------------------------------------
        // invalid bytemask (0x00 or 0xff)
        movi v0.2d, #0xffff00ff001f00ff

// CHECK:ERROR: error: invalid operand for instruction
// CHECK:ERROR:         movi v0.2d, #0xffff00ff001f00ff
// CHECK:ERROR:                     ^
//----------------------------------------------------------------------
// Vector Floating Point Move Immediate
//----------------------------------------------------------------------
        // invalid vector type (2s, 4s, 2d)
         fmov v0.4h, #1.0
         // invalid immediate (negative hexadecimal encoding)
         fmov v0.4s, #-0x0

// CHECK:ERROR: error: invalid operand for instruction
// CHECK:ERROR:         fmov v0.4h, #1.0
// CHECK:ERROR:              ^
// CHECK-ERROR: error: encoded floating point value out of range
// CHECK-ERROR:         fmov v0.4s, #-0x0
// CHECK-ERROR:                     ^

//----------------------------------------------------------------------
// Vector Move -  register
//----------------------------------------------------------------------
      // invalid vector type (8b, 16b)
      mov v0.2s, v31.8b
// CHECK:ERROR: error: invalid operand for instruction
// CHECK:ERROR:         mov v0.2s, v31.8b
// CHECK:ERROR:                ^

//----------------------------------------------------------------------
// Vector Absolute Difference and Accumulate (Signed, Unsigned)
//----------------------------------------------------------------------

        // Mismatched and invalid vector types (2d)
        saba v0.16b, v1.8b, v2.8b
        uaba v0.2d, v1.2d, v2.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         saba v0.16b, v1.8b, v2.8b
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uaba v0.2d, v1.2d, v2.2d
// CHECK-ERROR:                ^

//----------------------------------------------------------------------
// Vector Absolute Difference and Accumulate (Signed, Unsigned)
// Vector Absolute Difference (Signed, Unsigned)

        // Mismatched and invalid vector types (2d)
        uaba v0.16b, v1.8b, v2.8b
        saba v0.2d, v1.2d, v2.2d
        uabd v0.4s, v1.2s, v2.2s
        sabd v0.4h, v1.8h, v8.8h

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uaba v0.16b, v1.8b, v2.8b
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         saba v0.2d, v1.2d, v2.2d
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uabd v0.4s, v1.2s, v2.2s
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sabd v0.4h, v1.8h, v8.8h
// CHECK-ERROR:                        ^

//----------------------------------------------------------------------
// Vector Absolute Difference (Floating Point)
//----------------------------------------------------------------------
        // Mismatched and invalid vector types
        fabd v0.2s, v1.4s, v2.2d
        fabd v0.4h, v1.4h, v2.4h

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fabd v0.2s, v1.4s, v2.2d
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fabd v0.4h, v1.4h, v2.4h
// CHECK-ERROR:                 ^
//----------------------------------------------------------------------
// Vector Multiply (Polynomial)
//----------------------------------------------------------------------

        // Mismatched and invalid vector types
         pmul v0.8b, v1.8b, v2.16b
         pmul v0.2s, v1.2s, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         pmul v0.8b, v1.8b, v2.16b
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         pmul v0.2s, v1.2s, v2.2s
// CHECK-ERROR:                 ^

//----------------------------------------------------------------------
// Scalar Integer Add and Sub
//----------------------------------------------------------------------

      // Mismatched registers
         add d0, s1, d2
         sub s1, d1, d2

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         add d0, s1, d2
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sub s1, d1, d2
// CHECK-ERROR:             ^

//----------------------------------------------------------------------
// Vector Reciprocal Step (Floating Point)
//----------------------------------------------------------------------

        // Mismatched and invalid vector types
         frecps v0.4s, v1.2d, v2.4s
         frecps v0.8h, v1.8h, v2.8h

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        frecps v0.4s, v1.2d, v2.4s
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:        frecps v0.8h, v1.8h, v2.8h
// CHECK-ERROR:                  ^

//----------------------------------------------------------------------
// Vector Reciprocal Square Root Step (Floating Point)
//----------------------------------------------------------------------

        // Mismatched and invalid vector types
         frsqrts v0.2d, v1.2d, v2.2s
         frsqrts v0.4h, v1.4h, v2.4h

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        frsqrts v0.2d, v1.2d, v2.2s
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:        frsqrts v0.4h, v1.4h, v2.4h
// CHECK-ERROR:                   ^


//----------------------------------------------------------------------
// Vector Absolute Compare Mask Less Than Or Equal (Floating Point)
//----------------------------------------------------------------------

        // Mismatched and invalid vector types
        facge v0.2d, v1.2s, v2.2d
        facge v0.4h, v1.4h, v2.4h
        facle v0.8h, v1.4h, v2.4h

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        facge v0.2d, v1.2s, v2.2d
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:        facge v0.4h, v1.4h, v2.4h
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        facle v0.8h, v1.4h, v2.4h
// CHECK-ERROR:                 ^
//----------------------------------------------------------------------
// Vector Absolute Compare Mask Less Than (Floating Point)
//----------------------------------------------------------------------

        // Mismatched and invalid vector types
        facgt v0.2d, v1.2d, v2.4s
        facgt v0.8h, v1.8h, v2.8h
        faclt v0.8b, v1.8b, v2.8b

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        facgt v0.2d, v1.2d, v2.4s
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:        facgt v0.8h, v1.8h, v2.8h
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        faclt v0.8b, v1.8b, v2.8b
// CHECK-ERROR:                 ^


//----------------------------------------------------------------------
// Vector Compare Mask Equal (Integer)
//----------------------------------------------------------------------

         // Mismatched vector types
         cmeq c0.2d, v1.2d, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        cmeq c0.2d, v1.2d, v2.2s
// CHECK-ERROR:                              ^

//----------------------------------------------------------------------
// Vector Compare Mask Higher or Same (Unsigned Integer)
// Vector Compare Mask Less or Same (Unsigned Integer)
// CMLS is alias for CMHS with operands reversed.
//----------------------------------------------------------------------

         // Mismatched vector types
         cmhs c0.4h, v1.8b, v2.8b
         cmls c0.16b, v1.16b, v2.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        cmhs c0.4h, v1.8b, v2.8b
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        cmls c0.16b, v1.16b, v2.2d
// CHECK-ERROR:                                ^

//----------------------------------------------------------------------
// Vector Compare Mask Greater Than or Equal (Integer)
// Vector Compare Mask Less Than or Equal (Integer)
// CMLE is alias for CMGE with operands reversed.
//----------------------------------------------------------------------

         // Mismatched vector types
         cmge c0.8h, v1.8b, v2.8b
         cmle c0.4h, v1.2s, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        cmge c0.8h, v1.8b, v2.8b
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         cmle c0.4h, v1.2s, v2.2s
// CHECK-ERROR:                        ^

//----------------------------------------------------------------------
// Vector Compare Mask Higher (Unsigned Integer)
// Vector Compare Mask Lower (Unsigned Integer)
// CMLO is alias for CMHI with operands reversed.
//----------------------------------------------------------------------

         // Mismatched vector types
         cmhi c0.4s, v1.4s, v2.16b
         cmlo c0.8b, v1.8b, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        cmhi c0.4s, v1.4s, v2.16b
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         cmlo c0.8b, v1.8b, v2.2s
// CHECK-ERROR:                               ^

//----------------------------------------------------------------------
// Vector Compare Mask Greater Than (Integer)
// Vector Compare Mask Less Than (Integer)
// CMLT is alias for CMGT with operands reversed.
//----------------------------------------------------------------------

         // Mismatched vector types
         cmgt c0.8b, v1.4s, v2.16b
         cmlt c0.8h, v1.16b, v2.4s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         cmgt c0.8b, v1.4s, v2.16b
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         cmlt c0.8h, v1.16b, v2.4s
// CHECK-ERROR:                        ^

//----------------------------------------------------------------------
// Vector Compare Mask Bitwise Test (Integer)
//----------------------------------------------------------------------

         // Mismatched vector types
         cmtst c0.16b, v1.16b, v2.4s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         cmtst c0.16b, v1.16b, v2.4s
// CHECK-ERROR:                                  ^

//----------------------------------------------------------------------
// Vector Compare Mask Equal (Floating Point)
//----------------------------------------------------------------------

        // Mismatched and invalid vector types
        fcmeq v0.2d, v1.2s, v2.2d
        fcmeq v0.16b, v1.16b, v2.16b
        fcmeq v0.8b, v1.4h, v2.4h

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmeq v0.2d, v1.2s, v2.2d
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmeq v0.16b, v1.16b, v2.16b
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmeq v0.8b, v1.4h, v2.4h
// CHECK-ERROR:                 ^

//----------------------------------------------------------------------
// Vector Compare Mask Greater Than Or Equal (Floating Point)
// Vector Compare Mask Less Than Or Equal (Floating Point)
// FCMLE is alias for FCMGE with operands reversed.
//----------------------------------------------------------------------

        // Mismatched and invalid vector types
         fcmge v31.4s, v29.2s, v28.4s
         fcmge v3.8b, v8.2s, v12.2s
         fcmle v17.8h, v15.2d, v13.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmge v31.4s, v29.2s, v28.4s
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmge v3.8b, v8.2s, v12.2s
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmle v17.8h, v15.2d, v13.2d
// CHECK-ERROR:                 ^

//----------------------------------------------------------------------
// Vector Compare Mask Greater Than (Floating Point)
// Vector Compare Mask Less Than (Floating Point)
// FCMLT is alias for FCMGT with operands reversed.
//----------------------------------------------------------------------

        // Mismatched and invalid vector types
         fcmgt v0.2d, v31.2s, v16.2s
         fcmgt v4.4s, v7.4s, v15.4h
         fcmlt v29.2d, v5.2d, v2.16b

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmgt v0.2d, v31.2s, v16.2s
// CHECK-ERROR:                         ^

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmgt v4.4s, v7.4s, v15.4h
// CHECK-ERROR:                                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmlt v29.2d, v5.2d, v2.16b
// CHECK-ERROR:                                ^

//----------------------------------------------------------------------
// Vector Compare Mask Equal to Zero (Integer)
//----------------------------------------------------------------------
        // Mismatched vector types and invalid imm
         // Mismatched vector types
         cmeq c0.2d, v1.2s, #0
         cmeq c0.2d, v1.2d, #1

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        cmeq c0.2d, v1.2s, #0
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        cmeq c0.2d, v1.2d, #1
// CHECK-ERROR:                            ^

//----------------------------------------------------------------------
// Vector Compare Mask Greater Than or Equal to Zero (Signed Integer)
//----------------------------------------------------------------------
        // Mismatched vector types and invalid imm
         cmge c0.8h, v1.8b, #0
         cmge c0.4s, v1.4s, #-1

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        cmge c0.8h, v1.8b, #0
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         cmge c0.4s, v1.4s, #-1
// CHECK-ERROR:                             ^

//----------------------------------------------------------------------
// Vector Compare Mask Greater Than Zero (Signed Integer)
//----------------------------------------------------------------------
        // Mismatched vector types and invalid imm
         cmgt c0.8b, v1.4s, #0
         cmgt c0.8b, v1.8b, #-255

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         cmgt c0.8b, v1.4s, #0
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         cmgt c0.8b, v1.8b, #-255
// CHECK-ERROR:                             ^

//----------------------------------------------------------------------
// Vector Compare Mask Less Than or Equal To Zero (Signed Integer)
//----------------------------------------------------------------------
        // Mismatched vector types and invalid imm
         cmle c0.4h, v1.2s, #0
         cmle c0.16b, v1.16b, #16

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        cmle c0.4h, v1.2s, #0
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         cmle c0.16b, v1.16b, #16
// CHECK-ERROR:                               ^
//----------------------------------------------------------------------
// Vector Compare Mask Less Than Zero (Signed Integer)
//----------------------------------------------------------------------
        // Mismatched vector types and invalid imm
         cmlt c0.8h, v1.16b, #0
         cmlt c0.8h, v1.8h, #-15

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         cmlt c0.8h, v1.16b, #0
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         cmlt c0.8h, v1.8h, #-15
// CHECK-ERROR:                             ^

//----------------------------------------------------------------------
// Vector Compare Mask Equal to Zero (Floating Point)
//----------------------------------------------------------------------

        // Mismatched and invalid vector types, invalid imm
        fcmeq v0.2d, v1.2s, #0.0
        fcmeq v0.16b, v1.16b, #0.0
        fcmeq v0.8b, v1.4h, #1.0
        fcmeq v0.8b, v1.4h, #1

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmeq v0.2d, v1.2s, #0.0
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmeq v0.16b, v1.16b, #0.0
// CHECK-ERROR:                 ^


// CHECK-ERROR: error: expected floating-point constant #0.0
// CHECK-ERROR:        fcmeq v0.8b, v1.4h, #1.0
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmeq v0.8b, v1.4h, #1
// CHECK-ERROR:                             ^

//----------------------------------------------------------------------
// Vector Compare Mask Greater Than or Equal to Zero (Floating Point)
//----------------------------------------------------------------------

        // Mismatched and invalid vector types, invalid imm
         fcmge v31.4s, v29.2s, #0.0
         fcmge v3.8b, v8.2s, #0.0
         fcmle v17.8h, v15.2d, #-1.0
         fcmle v17.8h, v15.2d, #2

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmge v31.4s, v29.2s, #0.0
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmge v3.8b, v8.2s, #0.0
// CHECK-ERROR:                 ^


// CHECK-ERROR: error: expected floating-point constant #0.0
// CHECK-ERROR:        fcmle v17.8h, v15.2d, #-1.0
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmle v17.8h, v15.2d, #2
// CHECK-ERROR:                               ^

//----------------------------------------------------------------------
// Vector Compare Mask Greater Than Zero (Floating Point)
//----------------------------------------------------------------------
        // Mismatched and invalid vector types, invalid imm
         fcmgt v0.2d, v31.2s, #0.0
         fcmgt v4.4s, v7.4h, #0.0
         fcmlt v29.2d, v5.2d, #255.0
         fcmlt v29.2d, v5.2d, #255

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmgt v0.2d, v31.2s, #0.0
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmgt v4.4s, v7.4h, #0.0
// CHECK-ERROR:                        ^


// CHECK-ERROR: error: expected floating-point constant #0.0
// CHECK-ERROR:        fcmlt v29.2d, v5.2d, #255.0
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmlt v29.2d, v5.2d, #255
// CHECK-ERROR:                              ^

//----------------------------------------------------------------------
// Vector Compare Mask Less Than or Equal To Zero (Floating Point)
//----------------------------------------------------------------------
        // Mismatched and invalid vector types, invalid imm
         fcmge v31.4s, v29.2s, #0.0
         fcmge v3.8b, v8.2s, #0.0
         fcmle v17.2d, v15.2d, #15.0
         fcmle v17.2d, v15.2d, #15

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmge v31.4s, v29.2s, #0.0
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmge v3.8b, v8.2s, #0.0
// CHECK-ERROR:                 ^


// CHECK-ERROR: error: expected floating-point constant #0.0
// CHECK-ERROR:        fcmle v17.2d, v15.2d, #15.0
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmle v17.2d, v15.2d, #15
// CHECK-ERROR:                              ^

//----------------------------------------------------------------------
// Vector Compare Mask Less Than Zero (Floating Point)
//----------------------------------------------------------------------
        // Mismatched and invalid vector types, invalid imm
         fcmgt v0.2d, v31.2s, #0.0
         fcmgt v4.4s, v7.4h, #0.0
         fcmlt v29.2d, v5.2d, #16.0
         fcmlt v29.2d, v5.2d, #2

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmgt v0.2d, v31.2s, #0.0
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmgt v4.4s, v7.4h, #0.0
// CHECK-ERROR:                        ^


// CHECK-ERROR: error: expected floating-point constant #0.0
// CHECK-ERROR:        fcmlt v29.2d, v5.2d, #16.0
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcmlt v29.2d, v5.2d, #2
// CHECK-ERROR:                              ^

/-----------------------------------------------------------------------
// Vector Integer Halving Add (Signed)
// Vector Integer Halving Add (Unsigned)
// Vector Integer Halving Sub (Signed)
// Vector Integer Halving Sub (Unsigned)
//----------------------------------------------------------------------
        // Mismatched and invalid vector types (2d)
        shadd v0.2d, v1.2d, v2.2d
        uhadd v4.2s, v5.2s, v5.4h
        shsub v11.4h, v12.8h, v13.4h
        uhsub v31.16b, v29.8b, v28.8b

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        shadd v0.2d, v1.2d, v2.2d
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uhadd v4.2s, v5.2s, v5.4h
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        shsub v11.4h, v12.8h, v13.4h
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uhsub v31.16b, v29.8b, v28.8b
// CHECK-ERROR:                          ^

//----------------------------------------------------------------------
// Vector Integer Rouding Halving Add (Signed)
// Vector Integer Rouding Halving Add (Unsigned)
//----------------------------------------------------------------------

        // Mismatched and invalid vector types (2d)
        srhadd v0.2s, v1.2s, v2.2d
        urhadd v0.16b, v1.16b, v2.8h

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        srhadd v0.2s, v1.2s, v2.2d
// CHECK-ERROR:                                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        urhadd v0.16b, v1.16b, v2.8h
// CHECK-ERROR:                                  ^

//----------------------------------------------------------------------
// Vector Integer Saturating Add (Signed)
// Vector Integer Saturating Add (Unsigned)
// Vector Integer Saturating Sub (Signed)
// Vector Integer Saturating Sub (Unsigned)
//----------------------------------------------------------------------

        // Mismatched vector types
        sqadd v0.2s, v1.2s, v2.2d
        uqadd v31.8h, v1.4h, v2.4h
        sqsub v10.8h, v1.16b, v2.16b
        uqsub v31.8b, v1.8b, v2.4s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqadd v0.2s, v1.2s, v2.2d
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uqadd v31.8h, v1.4h, v2.4h
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqsub v10.8h, v1.16b, v2.16b
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uqsub v31.8b, v1.8b, v2.4s
// CHECK-ERROR:                                ^

//----------------------------------------------------------------------
// Scalar Integer Saturating Add (Signed)
// Scalar Integer Saturating Add (Unsigned)
// Scalar Integer Saturating Sub (Signed)
// Scalar Integer Saturating Sub (Unsigned)
//----------------------------------------------------------------------

      // Mismatched registers
         sqadd d0, s31, d2
         uqadd s0, s1, d2
         sqsub b0, b2, s18
         uqsub h1, h2, d2

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqadd d0, s31, d2
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uqadd s0, s1, d2
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqsub b0, b2, s18
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uqsub h1, h2, d2
// CHECK-ERROR:                      ^

//----------------------------------------------------------------------
// Scalar Integer Saturating Doubling Multiply Half High (Signed)
//----------------------------------------------------------------------

    sqdmulh h10, s11, h12
    sqdmulh s20, h21, s2

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmulh h10, s11, h12
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmulh s20, h21, s2
// CHECK-ERROR:                     ^

//------------------------------------------------------------------------
// Scalar Integer Saturating Rounding Doubling Multiply Half High (Signed)
//------------------------------------------------------------------------

    sqrdmulh h10, s11, h12
    sqrdmulh s20, h21, s2

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqrdmulh h10, s11, h12
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqrdmulh s20, h21, s2
// CHECK-ERROR:                      ^

//----------------------------------------------------------------------
// Vector Shift Left (Signed and Unsigned Integer)
//----------------------------------------------------------------------
        // Mismatched vector types
        sshl v0.4s, v15.2s, v16.2s
        ushl v1.16b, v25.16b, v6.8h

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sshl v0.4s, v15.2s, v16.2s
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        ushl v1.16b, v25.16b, v6.8h
// CHECK-ERROR:                                 ^

//----------------------------------------------------------------------
// Vector Saturating Shift Left (Signed and Unsigned Integer)
//----------------------------------------------------------------------
        // Mismatched vector types
        sqshl v0.2s, v15.4s, v16.2d
        uqshl v1.8b, v25.4h, v6.8h

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqshl v0.2s, v15.4s, v16.2d 
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uqshl v1.8b, v25.4h, v6.8h
// CHECK-ERROR:                         ^

//----------------------------------------------------------------------
// Vector Rouding Shift Left (Signed and Unsigned Integer)
//----------------------------------------------------------------------
        // Mismatched vector types
        srshl v0.8h, v15.8h, v16.16b
        urshl v1.2d, v25.2d, v6.4s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        srshl v0.8h, v15.8h, v16.16b
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        urshl v1.2d, v25.2d, v6.4s
// CHECK-ERROR:                                ^

//----------------------------------------------------------------------
// Vector Saturating Rouding Shift Left (Signed and Unsigned Integer)
//----------------------------------------------------------------------
        // Mismatched vector types
        sqrshl v0.2s, v15.8h, v16.16b
        uqrshl v1.4h, v25.4h,  v6.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqrshl v0.2s, v15.8h, v16.16b
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uqrshl v1.4h, v25.4h,  v6.2d
// CHECK-ERROR:                                  ^

//----------------------------------------------------------------------
// Scalar Integer Shift Left (Signed, Unsigned)
//----------------------------------------------------------------------
        // Mismatched and invalid vector types
        sshl d0, d1, s2
        ushl b2, b0, b1

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sshl d0, d1, s2
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        ushl b2, b0, b1
// CHECK-ERROR:             ^

//----------------------------------------------------------------------
// Scalar Integer Saturating Shift Left (Signed, Unsigned)
//----------------------------------------------------------------------

        // Mismatched vector types
        sqshl b0, s1, b0
        uqshl h0, b1, h0
        sqshl s0, h1, s0
        uqshl d0, b1, d0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqshl b0, s1, b0
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uqshl h0, b1, h0
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqshl s0, h1, s0
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uqshl d0, b1, d0
// CHECK-ERROR:                  ^

//----------------------------------------------------------------------
// Scalar Integer Rouding Shift Left (Signed, Unsigned)
//----------------------------------------------------------------------
        // Mismatched and invalid vector types
        srshl h0, h1, h2
        urshl s0, s1, s2

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        srshl h0, h1, h2
// CHECK-ERROR:              ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        urshl s0, s1, s2
// CHECK-ERROR:              ^


//----------------------------------------------------------------------
// Scalar Integer Saturating Rounding Shift Left (Signed, Unsigned)
//----------------------------------------------------------------------

        // Mismatched vector types
        sqrshl b0, b1, s0
        uqrshl h0, h1, b0
        sqrshl s0, s1, h0
        uqrshl d0, d1, b0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqrshl b0, b1, s0
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uqrshl h0, h1, b0
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqrshl s0, s1, h0
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uqrshl d0, d1, b0
// CHECK-ERROR:                       ^


//----------------------------------------------------------------------
// Vector Maximum (Signed, Unsigned)
//----------------------------------------------------------------------
        // Mismatched and invalid vector types
        smax v0.2d, v1.2d, v2.2d
        umax v0.4h, v1.4h, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smax v0.2d, v1.2d, v2.2d
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umax v0.4h, v1.4h, v2.2s
// CHECK-ERROR:                              ^

//----------------------------------------------------------------------
// Vector Minimum (Signed, Unsigned)
//----------------------------------------------------------------------
        // Mismatched and invalid vector types
        smin v0.2d, v1.2d, v2.2d
        umin v0.2s, v1.2s, v2.8b

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smin v0.2d, v1.2d, v2.2d
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umin v0.2s, v1.2s, v2.8b
// CHECK-ERROR:                             ^


//----------------------------------------------------------------------
// Vector Maximum (Floating Point)
//----------------------------------------------------------------------
        // Mismatched and invalid vector types
        fmax v0.2s, v1.2s, v2.4s
        fmax v0.8b, v1.8b, v2.8b

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fmax v0.2s, v1.2s, v2.4s
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fmax v0.8b, v1.8b, v2.8b
// CHECK-ERROR:                ^
//----------------------------------------------------------------------
// Vector Minimum (Floating Point)
//----------------------------------------------------------------------
        // Mismatched and invalid vector types
        fmin v0.4s, v1.4s, v2.2d
        fmin v0.8h, v1.8h, v2.8h

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fmin v0.4s, v1.4s, v2.2d
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:        fmin v0.8h, v1.8h, v2.8h
// CHECK-ERROR:                ^

//----------------------------------------------------------------------
// Vector maxNum (Floating Point)
//----------------------------------------------------------------------
        // Mismatched and invalid vector types
        fmaxnm v0.2s, v1.2s, v2.2d
        fmaxnm v0.4h, v1.8h, v2.4h

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fmaxnm v0.2s, v1.2s, v2.2d
// CHECK-ERROR:                                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fmaxnm v0.4h, v1.8h, v2.4h
// CHECK-ERROR:                  ^

//----------------------------------------------------------------------
// Vector minNum (Floating Point)
//----------------------------------------------------------------------
        // Mismatched and invalid vector types
        fminnm v0.4s, v1.2s, v2.4s
        fminnm v0.16b, v0.16b, v0.16b

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fminnm v0.4s, v1.2s, v2.4s
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fminnm v0.16b, v0.16b, v0.16b
// CHECK-ERROR:                  ^


//----------------------------------------------------------------------
// Vector Maximum Pairwise (Signed, Unsigned)
//----------------------------------------------------------------------
        // Mismatched and invalid vector types
        smaxp v0.2d, v1.2d, v2.2d
        umaxp v0.4h, v1.4h, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smaxp v0.2d, v1.2d, v2.2d
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umaxp v0.4h, v1.4h, v2.2s
// CHECK-ERROR:                               ^

//----------------------------------------------------------------------
// Vector Minimum Pairwise (Signed, Unsigned)
//----------------------------------------------------------------------
        // Mismatched and invalid vector types
        sminp v0.2d, v1.2d, v2.2d
        uminp v0.2s, v1.2s, v2.8b

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sminp v0.2d, v1.2d, v2.2d
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uminp v0.2s, v1.2s, v2.8b
// CHECK-ERROR:                               ^


//----------------------------------------------------------------------
// Vector Maximum Pairwise (Floating Point)
//----------------------------------------------------------------------
        // Mismatched and invalid vector types
        fmaxp v0.2s, v1.2s, v2.4s
        fmaxp v0.8b, v1.8b, v2.8b

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fmaxp v0.2s, v1.2s, v2.4s
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fmaxp v0.8b, v1.8b, v2.8b
// CHECK-ERROR:                 ^
//----------------------------------------------------------------------
// Vector Minimum Pairwise (Floating Point)
//----------------------------------------------------------------------
        // Mismatched and invalid vector types
        fminp v0.4s, v1.4s, v2.2d
        fminp v0.8h, v1.8h, v2.8h

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fminp v0.4s, v1.4s, v2.2d
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:        fminp v0.8h, v1.8h, v2.8h
// CHECK-ERROR:                 ^

//----------------------------------------------------------------------
// Vector maxNum Pairwise (Floating Point)
//----------------------------------------------------------------------
        // Mismatched and invalid vector types
        fmaxnmp v0.2s, v1.2s, v2.2d
        fmaxnmp v0.4h, v1.8h, v2.4h

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fmaxnmp v0.2s, v1.2s, v2.2d
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fmaxnmp v0.4h, v1.8h, v2.4h
// CHECK-ERROR:                   ^

//----------------------------------------------------------------------
// Vector minNum Pairwise (Floating Point)
//----------------------------------------------------------------------
        // Mismatched and invalid vector types
        fminnmp v0.4s, v1.2s, v2.4s
        fminnmp v0.16b, v0.16b, v0.16b

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fminnmp v0.4s, v1.2s, v2.4s
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fminnmp v0.16b, v0.16b, v0.16b
// CHECK-ERROR:                   ^


//----------------------------------------------------------------------
// Vector Add Pairwise (Integer)
//----------------------------------------------------------------------

        // Mismatched vector types
        addp v0.16b, v1.8b, v2.8b

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         addp v0.16b, v1.8b, v2.8b
// CHECK-ERROR:                         ^

//----------------------------------------------------------------------
// Vector Add Pairwise (Floating Point)
//----------------------------------------------------------------------
        // Mismatched and invalid vector types
        faddp v0.16b, v1.8b, v2.8b
        faddp v0.2d, v1.2d, v2.8h

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         faddp v0.16b, v1.8b, v2.8b
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         faddp v0.2d, v1.2d, v2.8h
// CHECK-ERROR:                                ^


//----------------------------------------------------------------------
// Vector Saturating Doubling Multiply High
//----------------------------------------------------------------------
         // Mismatched and invalid vector types
         sqdmulh v2.4h, v25.8h, v3.4h
         sqdmulh v12.2d, v5.2d, v13.2d
         sqdmulh v3.8b, v1.8b, v30.8b

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqdmulh v2.4h, v25.8h, v3.4h
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqdmulh v12.2d, v5.2d, v13.2d
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqdmulh v3.8b, v1.8b, v30.8b
// CHECK-ERROR:                    ^

//----------------------------------------------------------------------
// Vector Saturating Rouding Doubling Multiply High
//----------------------------------------------------------------------
         // Mismatched and invalid vector types
         sqrdmulh v2.2s, v25.4s, v3.4s
         sqrdmulh v12.16b, v5.16b, v13.16b
         sqrdmulh v3.4h, v1.4h, v30.2d


// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqrdmulh v2.2s, v25.4s, v3.4s
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqrdmulh v12.16b, v5.16b, v13.16b
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqrdmulh v3.4h, v1.4h, v30.2d
// CHECK-ERROR:                                    ^

//----------------------------------------------------------------------
// Vector Multiply Extended
//----------------------------------------------------------------------
         // Mismatched and invalid vector types
      fmulx v21.2s, v5.2s, v13.2d
      fmulx v1.4h, v25.4h, v3.4h

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fmulx v21.2s, v5.2s, v13.2d
// CHECK-ERROR:                                  ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fmulx v1.4h, v25.4h, v3.4h
// CHECK-ERROR:                  ^

//------------------------------------------------------------------------------
// Vector Shift Left by Immediate
//------------------------------------------------------------------------------
         // Mismatched vector types and out of range
         shl v0.4s, v15,2s, #3
         shl v0.2d, v17.4s, #3
         shl v0.8b, v31.8b, #-1
         shl v0.8b, v31.8b, #8
         shl v0.4s, v21.4s, #32
         shl v0.2d, v1.2d, #64


// CHECK-ERROR: error: unexpected token in argument list
// CHECK-ERROR:         shl v0.4s, v15,2s, #3
// CHECK-ERROR:                         ^

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         shl v0.2d, v17.4s, #3
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 7]
// CHECK-ERROR:         shl v0.8b, v31.8b, #-1
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 7]
// CHECK-ERROR:         shl v0.8b, v31.8b, #8
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR:         shl v0.4s, v21.4s, #32
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR:         shl v0.2d, v1.2d, #64
// CHECK-ERROR:                           ^

//----------------------------------------------------------------------
// Vector Shift Left Long by Immediate
//----------------------------------------------------------------------
        // Mismatched vector types
        sshll v0.4s, v15.2s, #3
        ushll v1.16b, v25.16b, #6
        sshll2 v0.2d, v3.8s, #15
        ushll2 v1.4s, v25.4s, #7

        // Out of range 
        sshll v0.8h, v1.8b, #-1
        sshll v0.8h, v1.8b, #9
        ushll v0.4s, v1.4h, #17
        ushll v0.2d, v1.2s, #33
        sshll2 v0.8h, v1.16b, #9
        sshll2 v0.4s, v1.8h, #17
        ushll2 v0.2d, v1.4s, #33

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sshll v0.4s, v15.2s, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        ushll v1.16b, v25.16b, #6
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: invalid vector kind qualifier
// CHECK-ERROR:        sshll2 v0.2d, v3.8s, #15
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        ushll2 v1.4s, v25.4s, #7
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 7]
// CHECK-ERROR:        sshll v0.8h, v1.8b, #-1
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 7]
// CHECK-ERROR:        sshll v0.8h, v1.8b, #9
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR:        ushll v0.4s, v1.4h, #17
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR:        ushll v0.2d, v1.2s, #33
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 7]
// CHECK-ERROR:        sshll2 v0.8h, v1.16b, #9
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR:        sshll2 v0.4s, v1.8h, #17
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR:        ushll2 v0.2d, v1.4s, #33
// CHECK-ERROR:                             ^


//------------------------------------------------------------------------------
// Vector shift right by immediate
//------------------------------------------------------------------------------
         sshr v0.8b, v1.8h, #3
         sshr v0.4h, v1.4s, #3
         sshr v0.2s, v1.2d, #3
         sshr v0.16b, v1.16b, #9
         sshr v0.8h, v1.8h, #17
         sshr v0.4s, v1.4s, #33
         sshr v0.2d, v1.2d, #65

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sshr v0.8b, v1.8h, #3
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sshr v0.4h, v1.4s, #3
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sshr v0.2s, v1.2d, #3
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:         sshr v0.16b, v1.16b, #9
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:         sshr v0.8h, v1.8h, #17
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:         sshr v0.4s, v1.4s, #33
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:         sshr v0.2d, v1.2d, #65
// CHECK-ERROR:                            ^

//------------------------------------------------------------------------------
// Vector  shift right by immediate
//------------------------------------------------------------------------------
         ushr v0.8b, v1.8h, #3
         ushr v0.4h, v1.4s, #3
         ushr v0.2s, v1.2d, #3
         ushr v0.16b, v1.16b, #9
         ushr v0.8h, v1.8h, #17
         ushr v0.4s, v1.4s, #33
         ushr v0.2d, v1.2d, #65

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ushr v0.8b, v1.8h, #3
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ushr v0.4h, v1.4s, #3
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ushr v0.2s, v1.2d, #3
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:         ushr v0.16b, v1.16b, #9
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:         ushr v0.8h, v1.8h, #17
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:         ushr v0.4s, v1.4s, #33
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:         ushr v0.2d, v1.2d, #65
// CHECK-ERROR:                            ^

//------------------------------------------------------------------------------
// Vector shift right and accumulate by immediate
//------------------------------------------------------------------------------
         ssra v0.8b, v1.8h, #3
         ssra v0.4h, v1.4s, #3
         ssra v0.2s, v1.2d, #3
         ssra v0.16b, v1.16b, #9
         ssra v0.8h, v1.8h, #17
         ssra v0.4s, v1.4s, #33
         ssra v0.2d, v1.2d, #65

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ssra v0.8b, v1.8h, #3
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ssra v0.4h, v1.4s, #3
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ssra v0.2s, v1.2d, #3
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:         ssra v0.16b, v1.16b, #9
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:         ssra v0.8h, v1.8h, #17
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:         ssra v0.4s, v1.4s, #33
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:         ssra v0.2d, v1.2d, #65
// CHECK-ERROR:                            ^

//------------------------------------------------------------------------------
// Vector  shift right and accumulate by immediate
//------------------------------------------------------------------------------
         usra v0.8b, v1.8h, #3
         usra v0.4h, v1.4s, #3
         usra v0.2s, v1.2d, #3
         usra v0.16b, v1.16b, #9
         usra v0.8h, v1.8h, #17
         usra v0.4s, v1.4s, #33
         usra v0.2d, v1.2d, #65

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         usra v0.8b, v1.8h, #3
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         usra v0.4h, v1.4s, #3
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         usra v0.2s, v1.2d, #3
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:         usra v0.16b, v1.16b, #9
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:         usra v0.8h, v1.8h, #17
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:         usra v0.4s, v1.4s, #33
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:         usra v0.2d, v1.2d, #65
// CHECK-ERROR:                            ^

//------------------------------------------------------------------------------
// Vector rounding shift right by immediate
//------------------------------------------------------------------------------
         srshr v0.8b, v1.8h, #3
         srshr v0.4h, v1.4s, #3
         srshr v0.2s, v1.2d, #3
         srshr v0.16b, v1.16b, #9
         srshr v0.8h, v1.8h, #17
         srshr v0.4s, v1.4s, #33
         srshr v0.2d, v1.2d, #65

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         srshr v0.8b, v1.8h, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         srshr v0.4h, v1.4s, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         srshr v0.2s, v1.2d, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:         srshr v0.16b, v1.16b, #9
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:         srshr v0.8h, v1.8h, #17
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:         srshr v0.4s, v1.4s, #33
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:         srshr v0.2d, v1.2d, #65
// CHECK-ERROR:                             ^

//------------------------------------------------------------------------------
// Vecotr rounding shift right by immediate
//------------------------------------------------------------------------------
         urshr v0.8b, v1.8h, #3
         urshr v0.4h, v1.4s, #3
         urshr v0.2s, v1.2d, #3
         urshr v0.16b, v1.16b, #9
         urshr v0.8h, v1.8h, #17
         urshr v0.4s, v1.4s, #33
         urshr v0.2d, v1.2d, #65

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         urshr v0.8b, v1.8h, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         urshr v0.4h, v1.4s, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         urshr v0.2s, v1.2d, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:         urshr v0.16b, v1.16b, #9
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:         urshr v0.8h, v1.8h, #17
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:         urshr v0.4s, v1.4s, #33
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:         urshr v0.2d, v1.2d, #65
// CHECK-ERROR:                             ^

//------------------------------------------------------------------------------
// Vector rounding shift right and accumulate by immediate
//------------------------------------------------------------------------------
         srsra v0.8b, v1.8h, #3
         srsra v0.4h, v1.4s, #3
         srsra v0.2s, v1.2d, #3
         srsra v0.16b, v1.16b, #9
         srsra v0.8h, v1.8h, #17
         srsra v0.4s, v1.4s, #33
         srsra v0.2d, v1.2d, #65

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         srsra v0.8b, v1.8h, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         srsra v0.4h, v1.4s, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         srsra v0.2s, v1.2d, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:         srsra v0.16b, v1.16b, #9
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:         srsra v0.8h, v1.8h, #17
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:         srsra v0.4s, v1.4s, #33
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:         srsra v0.2d, v1.2d, #65
// CHECK-ERROR:                             ^

//------------------------------------------------------------------------------
// Vector rounding shift right and accumulate by immediate
//------------------------------------------------------------------------------
         ursra v0.8b, v1.8h, #3
         ursra v0.4h, v1.4s, #3
         ursra v0.2s, v1.2d, #3
         ursra v0.16b, v1.16b, #9
         ursra v0.8h, v1.8h, #17
         ursra v0.4s, v1.4s, #33
         ursra v0.2d, v1.2d, #65

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ursra v0.8b, v1.8h, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ursra v0.4h, v1.4s, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ursra v0.2s, v1.2d, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:         ursra v0.16b, v1.16b, #9
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:         ursra v0.8h, v1.8h, #17
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:         ursra v0.4s, v1.4s, #33
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:         ursra v0.2d, v1.2d, #65
// CHECK-ERROR:                             ^

//------------------------------------------------------------------------------
// Vector shift right and insert by immediate
//------------------------------------------------------------------------------
         sri v0.8b, v1.8h, #3
         sri v0.4h, v1.4s, #3
         sri v0.2s, v1.2d, #3
         sri v0.16b, v1.16b, #9
         sri v0.8h, v1.8h, #17
         sri v0.4s, v1.4s, #33
         sri v0.2d, v1.2d, #65

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sri v0.8b, v1.8h, #3
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sri v0.4h, v1.4s, #3
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sri v0.2s, v1.2d, #3
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:         sri v0.16b, v1.16b, #9
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:         sri v0.8h, v1.8h, #17
// CHECK-ERROR:                           ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:         sri v0.4s, v1.4s, #33
// CHECK-ERROR:                           ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:         sri v0.2d, v1.2d, #65
// CHECK-ERROR:                           ^

//------------------------------------------------------------------------------
// Vector shift left and insert by immediate
//------------------------------------------------------------------------------
         sli v0.8b, v1.8h, #3
         sli v0.4h, v1.4s, #3
         sli v0.2s, v1.2d, #3
         sli v0.16b, v1.16b, #8
         sli v0.8h, v1.8h, #16
         sli v0.4s, v1.4s, #32
         sli v0.2d, v1.2d, #64

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sli v0.8b, v1.8h, #3
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sli v0.4h, v1.4s, #3
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sli v0.2s, v1.2d, #3
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 7]
// CHECK-ERROR:         sli v0.16b, v1.16b, #8
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR:         sli v0.8h, v1.8h, #16
// CHECK-ERROR:                           ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR:         sli v0.4s, v1.4s, #32
// CHECK-ERROR:                           ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR:         sli v0.2d, v1.2d, #64
// CHECK-ERROR:                           ^

//------------------------------------------------------------------------------
// Vector saturating shift left unsigned by immediate
//------------------------------------------------------------------------------
         sqshlu v0.8b, v1.8h, #3
         sqshlu v0.4h, v1.4s, #3
         sqshlu v0.2s, v1.2d, #3
         sqshlu v0.16b, v1.16b, #8
         sqshlu v0.8h, v1.8h, #16
         sqshlu v0.4s, v1.4s, #32
         sqshlu v0.2d, v1.2d, #64

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqshlu v0.8b, v1.8h, #3
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqshlu v0.4h, v1.4s, #3
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqshlu v0.2s, v1.2d, #3
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 7]
// CHECK-ERROR:         sqshlu v0.16b, v1.16b, #8
// CHECK-ERROR:                                ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR:         sqshlu v0.8h, v1.8h, #16
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR:         sqshlu v0.4s, v1.4s, #32
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR:         sqshlu v0.2d, v1.2d, #64
// CHECK-ERROR:                              ^

//------------------------------------------------------------------------------
// Vector saturating shift left by immediate
//------------------------------------------------------------------------------
         sqshl v0.8b, v1.8h, #3
         sqshl v0.4h, v1.4s, #3
         sqshl v0.2s, v1.2d, #3
         sqshl v0.16b, v1.16b, #8
         sqshl v0.8h, v1.8h, #16
         sqshl v0.4s, v1.4s, #32
         sqshl v0.2d, v1.2d, #64

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqshl v0.8b, v1.8h, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqshl v0.4h, v1.4s, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqshl v0.2s, v1.2d, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 7]
// CHECK-ERROR:         sqshl v0.16b, v1.16b, #8
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR:         sqshl v0.8h, v1.8h, #16
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR:         sqshl v0.4s, v1.4s, #32
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR:         sqshl v0.2d, v1.2d, #64
// CHECK-ERROR:                             ^

//------------------------------------------------------------------------------
// Vector saturating shift left by immediate
//------------------------------------------------------------------------------
         uqshl v0.8b, v1.8h, #3
         uqshl v0.4h, v1.4s, #3
         uqshl v0.2s, v1.2d, #3
         uqshl v0.16b, v1.16b, #8
         uqshl v0.8h, v1.8h, #16
         uqshl v0.4s, v1.4s, #32
         uqshl v0.2d, v1.2d, #64

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uqshl v0.8b, v1.8h, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uqshl v0.4h, v1.4s, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uqshl v0.2s, v1.2d, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 7]
// CHECK-ERROR:         uqshl v0.16b, v1.16b, #8
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR:         uqshl v0.8h, v1.8h, #16
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR:         uqshl v0.4s, v1.4s, #32
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR:         uqshl v0.2d, v1.2d, #64
// CHECK-ERROR:                             ^

//------------------------------------------------------------------------------
// Vector shift right narrow by immediate
//------------------------------------------------------------------------------
         shrn v0.8b, v1.8b, #3
         shrn v0.4h, v1.4h, #3
         shrn v0.2s, v1.2s, #3
         shrn2 v0.16b, v1.8h, #17
         shrn2 v0.8h, v1.4s, #33
         shrn2 v0.4s, v1.2d, #65

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         shrn v0.8b, v1.8b, #3
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         shrn v0.4h, v1.4h, #3
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         shrn v0.2s, v1.2s, #3
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:         shrn2 v0.16b, v1.8h, #17
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:         shrn2 v0.8h, v1.4s, #33
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:         shrn2 v0.4s, v1.2d, #65
// CHECK-ERROR:                             ^

//------------------------------------------------------------------------------
// Vector saturating shift right unsigned narrow by immediate
//------------------------------------------------------------------------------
         sqshrun v0.8b, v1.8b, #3
         sqshrun v0.4h, v1.4h, #3
         sqshrun v0.2s, v1.2s, #3
         sqshrun2 v0.16b, v1.8h, #17
         sqshrun2 v0.8h, v1.4s, #33
         sqshrun2 v0.4s, v1.2d, #65

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqshrun v0.8b, v1.8b, #3
// CHECK-ERROR:                           ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqshrun v0.4h, v1.4h, #3
// CHECK-ERROR:                           ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqshrun v0.2s, v1.2s, #3
// CHECK-ERROR:                           ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:         sqshrun2 v0.16b, v1.8h, #17
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:         sqshrun2 v0.8h, v1.4s, #33
// CHECK-ERROR:                                ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:         sqshrun2 v0.4s, v1.2d, #65
// CHECK-ERROR:                                ^

//------------------------------------------------------------------------------
// Vector rounding shift right narrow by immediate
//------------------------------------------------------------------------------
         rshrn v0.8b, v1.8b, #3
         rshrn v0.4h, v1.4h, #3
         rshrn v0.2s, v1.2s, #3
         rshrn2 v0.16b, v1.8h, #17
         rshrn2 v0.8h, v1.4s, #33
         rshrn2 v0.4s, v1.2d, #65

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         rshrn v0.8b, v1.8b, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         rshrn v0.4h, v1.4h, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         rshrn v0.2s, v1.2s, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:         rshrn2 v0.16b, v1.8h, #17
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:         rshrn2 v0.8h, v1.4s, #33
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:         rshrn2 v0.4s, v1.2d, #65
// CHECK-ERROR:                              ^

//------------------------------------------------------------------------------
// Vector saturating shift right rounded unsigned narrow by immediate
//------------------------------------------------------------------------------
         sqrshrun v0.8b, v1.8b, #3
         sqrshrun v0.4h, v1.4h, #3
         sqrshrun v0.2s, v1.2s, #3
         sqrshrun2 v0.16b, v1.8h, #17
         sqrshrun2 v0.8h, v1.4s, #33
         sqrshrun2 v0.4s, v1.2d, #65

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqrshrun v0.8b, v1.8b, #3
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqrshrun v0.4h, v1.4h, #3
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqrshrun v0.2s, v1.2s, #3
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:         sqrshrun2 v0.16b, v1.8h, #17
// CHECK-ERROR:                                  ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:         sqrshrun2 v0.8h, v1.4s, #33
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:         sqrshrun2 v0.4s, v1.2d, #65
// CHECK-ERROR:                                 ^

//------------------------------------------------------------------------------
// Vector saturating shift right narrow by immediate
//------------------------------------------------------------------------------
         sqshrn v0.8b, v1.8b, #3
         sqshrn v0.4h, v1.4h, #3
         sqshrn v0.2s, v1.2s, #3
         sqshrn2 v0.16b, v1.8h, #17
         sqshrn2 v0.8h, v1.4s, #33
         sqshrn2 v0.4s, v1.2d, #65

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqshrn v0.8b, v1.8b, #3
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqshrn v0.4h, v1.4h, #3
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqshrn v0.2s, v1.2s, #3
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:         sqshrn2 v0.16b, v1.8h, #17
// CHECK-ERROR:                                ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:         sqshrn2 v0.8h, v1.4s, #33
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:         sqshrn2 v0.4s, v1.2d, #65
// CHECK-ERROR:                               ^

//------------------------------------------------------------------------------
// Vector saturating shift right narrow by immediate
//------------------------------------------------------------------------------
         uqshrn v0.8b, v1.8b, #3
         uqshrn v0.4h, v1.4h, #3
         uqshrn v0.2s, v1.2s, #3
         uqshrn2 v0.16b, v1.8h, #17
         uqshrn2 v0.8h, v1.4s, #33
         uqshrn2 v0.4s, v1.2d, #65

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uqshrn v0.8b, v1.8b, #3
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uqshrn v0.4h, v1.4h, #3
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uqshrn v0.2s, v1.2s, #3
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:         uqshrn2 v0.16b, v1.8h, #17
// CHECK-ERROR:                                ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:         uqshrn2 v0.8h, v1.4s, #33
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:         uqshrn2 v0.4s, v1.2d, #65
// CHECK-ERROR:                               ^

//------------------------------------------------------------------------------
// Vector saturating shift right rounded narrow by immediate
//------------------------------------------------------------------------------
         sqrshrn v0.8b, v1.8b, #3
         sqrshrn v0.4h, v1.4h, #3
         sqrshrn v0.2s, v1.2s, #3
         sqrshrn2 v0.16b, v1.8h, #17
         sqrshrn2 v0.8h, v1.4s, #33
         sqrshrn2 v0.4s, v1.2d, #65

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqrshrn v0.8b, v1.8b, #3
// CHECK-ERROR:                           ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqrshrn v0.4h, v1.4h, #3
// CHECK-ERROR:                           ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqrshrn v0.2s, v1.2s, #3
// CHECK-ERROR:                           ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:         sqrshrn2 v0.16b, v1.8h, #17
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:         sqrshrn2 v0.8h, v1.4s, #33
// CHECK-ERROR:                                ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:         sqrshrn2 v0.4s, v1.2d, #65
// CHECK-ERROR:                                ^

//------------------------------------------------------------------------------
// Vector saturating shift right rounded narrow by immediate
//------------------------------------------------------------------------------
         uqrshrn v0.8b, v1.8b, #3
         uqrshrn v0.4h, v1.4h, #3
         uqrshrn v0.2s, v1.2s, #3
         uqrshrn2 v0.16b, v1.8h, #17
         uqrshrn2 v0.8h, v1.4s, #33
         uqrshrn2 v0.4s, v1.2d, #65

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uqrshrn v0.8b, v1.8b, #3
// CHECK-ERROR:                           ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uqrshrn v0.4h, v1.4h, #3
// CHECK-ERROR:                           ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uqrshrn v0.2s, v1.2s, #3
// CHECK-ERROR:                           ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:         uqrshrn2 v0.16b, v1.8h, #17
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:         uqrshrn2 v0.8h, v1.4s, #33
// CHECK-ERROR:                                ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:         uqrshrn2 v0.4s, v1.2d, #65
// CHECK-ERROR:                                ^

//------------------------------------------------------------------------------
// Fixed-point convert to floating-point
//------------------------------------------------------------------------------
         scvtf v0.2s, v1.2d, #3
         scvtf v0.4s, v1.4h, #3
         scvtf v0.2d, v1.2s, #3
         ucvtf v0.2s, v1.2s, #33
         ucvtf v0.4s, v1.4s, #33
         ucvtf v0.2d, v1.2d, #65

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         scvtf v0.2s, v1.2d, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         scvtf v0.4s, v1.4h, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         scvtf v0.2d, v1.2s, #3
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:         ucvtf v0.2s, v1.2s, #33
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:         ucvtf v0.4s, v1.4s, #33
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:         ucvtf v0.2d, v1.2d, #65
// CHECK-ERROR:                             ^

//------------------------------------------------------------------------------
// Floating-point convert to fixed-point
//------------------------------------------------------------------------------
         fcvtzs v0.2s, v1.2d, #3
         fcvtzs v0.4s, v1.4h, #3
         fcvtzs v0.2d, v1.2s, #3
         fcvtzu v0.2s, v1.2s, #33
         fcvtzu v0.4s, v1.4s, #33
         fcvtzu v0.2d, v1.2d, #65

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtzs v0.2s, v1.2d, #3
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtzs v0.4s, v1.4h, #3
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtzs v0.2d, v1.2s, #3
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:         fcvtzu v0.2s, v1.2s, #33
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:         fcvtzu v0.4s, v1.4s, #33
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:         fcvtzu v0.2d, v1.2d, #65
// CHECK-ERROR:                              ^

//----------------------------------------------------------------------
// Vector operation on 3 operands with different types
//----------------------------------------------------------------------

        // Mismatched and invalid vector types
        saddl v0.8h, v1.8h, v2.8b
        saddl v0.4s, v1.4s, v2.4h
        saddl v0.2d, v1.2d, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        saddl v0.8h, v1.8h, v2.8b
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        saddl v0.4s, v1.4s, v2.4h
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        saddl v0.2d, v1.2d, v2.2s
// CHECK-ERROR:                        ^

        saddl2 v0.4s, v1.8s, v2.8h
        saddl2 v0.8h, v1.16h, v2.16b
        saddl2 v0.2d, v1.4d, v2.4s

// CHECK-ERROR: error: invalid vector kind qualifier
// CHECK-ERROR:        saddl2 v0.4s, v1.8s, v2.8h
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid vector kind qualifier
// CHECK-ERROR:        saddl2 v0.8h, v1.16h, v2.16b
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid vector kind qualifier
// CHECK-ERROR:        saddl2 v0.2d, v1.4d, v2.4s
// CHECK-ERROR:                      ^

        uaddl v0.8h, v1.8h, v2.8b
        uaddl v0.4s, v1.4s, v2.4h
        uaddl v0.2d, v1.2d, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uaddl v0.8h, v1.8h, v2.8b
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uaddl v0.4s, v1.4s, v2.4h
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uaddl v0.2d, v1.2d, v2.2s
// CHECK-ERROR:                        ^

        uaddl2 v0.8h, v1.16h, v2.16b
        uaddl2 v0.4s, v1.8s, v2.8h
        uaddl2 v0.2d, v1.4d, v2.4s

// CHECK-ERROR: [[@LINE-4]]:23: error: invalid vector kind qualifier
// CHECK-ERROR-NEXT:        uaddl2 v0.8h, v1.16h, v2.16b
// CHECK-ERROR: [[@LINE-5]]:23: error: invalid vector kind qualifier
// CHECK-ERROR-NEXT:        uaddl2 v0.4s, v1.8s, v2.8h
// CHECK-ERROR: [[@LINE-6]]:23: error: invalid vector kind qualifier
// CHECK-ERROR-NEXT:        uaddl2 v0.2d, v1.4d, v2.4s

        ssubl v0.8h, v1.8h, v2.8b
        ssubl v0.4s, v1.4s, v2.4h
        ssubl v0.2d, v1.2d, v2.2s

// CHECK-ERROR: [[@LINE-4]]:22: error: invalid operand for instruction
// CHECK-ERROR-NEXT:        ssubl v0.8h, v1.8h, v2.8b
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        ssubl v0.4s, v1.4s, v2.4h
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        ssubl v0.2d, v1.2d, v2.2s
// CHECK-ERROR:                        ^

        ssubl2 v0.8h, v1.16h, v2.16b
        ssubl2 v0.4s, v1.8s, v2.8h
        ssubl2 v0.2d, v1.4d, v2.4s

// CHECK-ERROR: [[@LINE-4]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        ssubl2 v0.8h, v1.16h, v2.16b
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid vector kind qualifier
// CHECK-ERROR:        ssubl2 v0.4s, v1.8s, v2.8h
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid vector kind qualifier
// CHECK-ERROR:        ssubl2 v0.2d, v1.4d, v2.4s
// CHECK-ERROR:                      ^

        usubl v0.8h, v1.8h, v2.8b
        usubl v0.4s, v1.4s, v2.4h
        usubl v0.2d, v1.2d, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        usubl v0.8h, v1.8h, v2.8b
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        usubl v0.4s, v1.4s, v2.4h
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        usubl v0.2d, v1.2d, v2.2s
// CHECK-ERROR:                        ^

        usubl2 v0.8h, v1.16h, v2.16b
        usubl2 v0.4s, v1.8s, v2.8h
        usubl2 v0.2d, v1.4d, v2.4s

// CHECK-ERROR: [[@LINE-4]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        usubl2 v0.8h, v1.16h, v2.16b
// CHECK-ERROR: [[@LINE-5]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        usubl2 v0.4s, v1.8s, v2.8h
// CHECK-ERROR: [[@LINE-6]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        usubl2 v0.2d, v1.4d, v2.4s

        sabal v0.8h, v1.8h, v2.8b
        sabal v0.4s, v1.4s, v2.4h
        sabal v0.2d, v1.2d, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sabal v0.8h, v1.8h, v2.8b
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sabal v0.4s, v1.4s, v2.4h
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sabal v0.2d, v1.2d, v2.2s
// CHECK-ERROR:                        ^

        sabal2 v0.8h, v1.16h, v2.16b
        sabal2 v0.4s, v1.8s, v2.8h
        sabal2 v0.2d, v1.4d, v2.4s

// CHECK-ERROR: error: invalid vector kind qualifier
// CHECK-ERROR:        sabal2 v0.8h, v1.16h, v2.16b
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid vector kind qualifier
// CHECK-ERROR:        sabal2 v0.4s, v1.8s, v2.8h
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid vector kind qualifier
// CHECK-ERROR:        sabal2 v0.2d, v1.4d, v2.4s
// CHECK-ERROR:                      ^

        uabal v0.8h, v1.8h, v2.8b
        uabal v0.4s, v1.4s, v2.4h
        uabal v0.2d, v1.2d, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uabal v0.8h, v1.8h, v2.8b
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uabal v0.4s, v1.4s, v2.4h
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uabal v0.2d, v1.2d, v2.2s
// CHECK-ERROR:                        ^

        uabal2 v0.8h, v1.16h, v2.16b
        uabal2 v0.4s, v1.8s, v2.8h
        uabal2 v0.2d, v1.4d, v2.4s

// CHECK-ERROR: [[@LINE-4]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        uabal2 v0.8h, v1.16h, v2.16b
// CHECK-ERROR: [[@LINE-5]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        uabal2 v0.4s, v1.8s, v2.8h
// CHECK-ERROR: [[@LINE-6]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        uabal2 v0.2d, v1.4d, v2.4s

        sabdl v0.8h, v1.8h, v2.8b
        sabdl v0.4s, v1.4s, v2.4h
        sabdl v0.2d, v1.2d, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sabdl v0.8h, v1.8h, v2.8b
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sabdl v0.4s, v1.4s, v2.4h
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sabdl v0.2d, v1.2d, v2.2s
// CHECK-ERROR:                        ^

        sabdl2 v0.8h, v1.16h, v2.16b
        sabdl2 v0.4s, v1.8s, v2.8h
        sabdl2 v0.2d, v1.4d, v2.4s

// CHECK-ERROR: [[@LINE-4]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        sabdl2 v0.8h, v1.16h, v2.16b
// CHECK-ERROR: [[@LINE-5]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        sabdl2 v0.4s, v1.8s, v2.8h
// CHECK-ERROR: [[@LINE-6]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        sabdl2 v0.2d, v1.4d, v2.4s

        uabdl v0.8h, v1.8h, v2.8b
        uabdl v0.4s, v1.4s, v2.4h
        uabdl v0.2d, v1.2d, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uabdl v0.8h, v1.8h, v2.8b
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uabdl v0.4s, v1.4s, v2.4h
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uabdl v0.2d, v1.2d, v2.2s
// CHECK-ERROR:                        ^

        uabdl2 v0.8h, v1.16h, v2.16b
        uabdl2 v0.4s, v1.8s, v2.8h
        uabdl2 v0.2d, v1.4d, v2.4s

// CHECK-ERROR: [[@LINE-4]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        uabdl2 v0.8h, v1.16h, v2.16b
// CHECK-ERROR: [[@LINE-5]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        uabdl2 v0.4s, v1.8s, v2.8h
// CHECK-ERROR: [[@LINE-6]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        uabdl2 v0.2d, v1.4d, v2.4s

        smlal v0.8h, v1.8h, v2.8b
        smlal v0.4s, v1.4s, v2.4h
        smlal v0.2d, v1.2d, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smlal v0.8h, v1.8h, v2.8b
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smlal v0.4s, v1.4s, v2.4h
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smlal v0.2d, v1.2d, v2.2s
// CHECK-ERROR:                        ^

        smlal2 v0.8h, v1.16h, v2.16b
        smlal2 v0.4s, v1.8s, v2.8h
        smlal2 v0.2d, v1.4d, v2.4s

// CHECK-ERROR: [[@LINE-4]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        smlal2 v0.8h, v1.16h, v2.16b
// CHECK-ERROR: [[@LINE-5]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        smlal2 v0.4s, v1.8s, v2.8h
// CHECK-ERROR: [[@LINE-6]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        smlal2 v0.2d, v1.4d, v2.4s

        umlal v0.8h, v1.8h, v2.8b
        umlal v0.4s, v1.4s, v2.4h
        umlal v0.2d, v1.2d, v2.2s

// CHECK-ERROR: [[@LINE-4]]:22: error: invalid operand for instruction
// CHECK-ERROR:        umlal v0.8h, v1.8h, v2.8b
// CHECK-ERROR: [[@LINE-5]]:22: error: invalid operand for instruction
// CHECK-ERROR:        umlal v0.4s, v1.4s, v2.4h
// CHECK-ERROR: [[@LINE-6]]:22: error: invalid operand for instruction
// CHECK-ERROR:        umlal v0.2d, v1.2d, v2.2s

        umlal2 v0.8h, v1.16h, v2.16b
        umlal2 v0.4s, v1.8s, v2.8h
        umlal2 v0.2d, v1.4d, v2.4s

// CHECK-ERROR: [[@LINE-4]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        umlal2 v0.8h, v1.16h, v2.16b
// CHECK-ERROR: [[@LINE-5]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        umlal2 v0.4s, v1.8s, v2.8h
// CHECK-ERROR: [[@LINE-6]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        umlal2 v0.2d, v1.4d, v2.4s

        smlsl v0.8h, v1.8h, v2.8b
        smlsl v0.4s, v1.4s, v2.4h
        smlsl v0.2d, v1.2d, v2.2s

// CHECK-ERROR: [[@LINE-4]]:22: error: invalid operand for instruction
// CHECK-ERROR:        smlsl v0.8h, v1.8h, v2.8b
// CHECK-ERROR: [[@LINE-5]]:22: error: invalid operand for instruction
// CHECK-ERROR:        smlsl v0.4s, v1.4s, v2.4h
// CHECK-ERROR: [[@LINE-6]]:22: error: invalid operand for instruction
// CHECK-ERROR:        smlsl v0.2d, v1.2d, v2.2s

        smlsl2 v0.8h, v1.16h, v2.16b
        smlsl2 v0.4s, v1.8s, v2.8h
        smlsl2 v0.2d, v1.4d, v2.4s

// CHECK-ERROR: [[@LINE-4]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        smlsl2 v0.8h, v1.16h, v2.16b
// CHECK-ERROR: [[@LINE-5]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        smlsl2 v0.4s, v1.8s, v2.8h
// CHECK-ERROR: [[@LINE-6]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        smlsl2 v0.2d, v1.4d, v2.4s

        umlsl v0.8h, v1.8h, v2.8b
        umlsl v0.4s, v1.4s, v2.4h
        umlsl v0.2d, v1.2d, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umlsl v0.8h, v1.8h, v2.8b
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umlsl v0.4s, v1.4s, v2.4h
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umlsl v0.2d, v1.2d, v2.2s
// CHECK-ERROR:                        ^

        umlsl2 v0.8h, v1.16h, v2.16b
        umlsl2 v0.4s, v1.8s, v2.8h
        umlsl2 v0.2d, v1.4d, v2.4s

// CHECK-ERROR: [[@LINE-4]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        umlsl2 v0.8h, v1.16h, v2.16b
// CHECK-ERROR: [[@LINE-5]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        umlsl2 v0.4s, v1.8s, v2.8h
// CHECK-ERROR: [[@LINE-6]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        umlsl2 v0.2d, v1.4d, v2.4s

        smull v0.8h, v1.8h, v2.8b
        smull v0.4s, v1.4s, v2.4h
        smull v0.2d, v1.2d, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smull v0.8h, v1.8h, v2.8b
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smull v0.4s, v1.4s, v2.4h
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smull v0.2d, v1.2d, v2.2s
// CHECK-ERROR:                        ^

        smull2 v0.8h, v1.16h, v2.16b
        smull2 v0.4s, v1.8s, v2.8h
        smull2 v0.2d, v1.4d, v2.4s

// CHECK-ERROR: [[@LINE-4]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        smull2 v0.8h, v1.16h, v2.16b
// CHECK-ERROR: [[@LINE-5]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        smull2 v0.4s, v1.8s, v2.8h
// CHECK-ERROR: [[@LINE-6]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        smull2 v0.2d, v1.4d, v2.4s

        umull v0.8h, v1.8h, v2.8b
        umull v0.4s, v1.4s, v2.4h
        umull v0.2d, v1.2d, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umull v0.8h, v1.8h, v2.8b
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umull v0.4s, v1.4s, v2.4h
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umull v0.2d, v1.2d, v2.2s
// CHECK-ERROR:                        ^

        umull2 v0.8h, v1.16h, v2.16b
        umull2 v0.4s, v1.8s, v2.8h
        umull2 v0.2d, v1.4d, v2.4s

// CHECK-ERROR: [[@LINE-4]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        umull2 v0.8h, v1.16h, v2.16b
// CHECK-ERROR: [[@LINE-5]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        umull2 v0.4s, v1.8s, v2.8h
// CHECK-ERROR: [[@LINE-6]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        umull2 v0.2d, v1.4d, v2.4s

//------------------------------------------------------------------------------
// Long - Variant 2
//------------------------------------------------------------------------------

        sqdmlal v0.4s, v1.4s, v2.4h
        sqdmlal v0.2d, v1.2d, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlal v0.4s, v1.4s, v2.4h
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlal v0.2d, v1.2d, v2.2s
// CHECK-ERROR:                          ^

        sqdmlal2 v0.4s, v1.8s, v2.8h
        sqdmlal2 v0.2d, v1.4d, v2.4s

// CHECK-ERROR: [[@LINE-3]]:25: error: invalid vector kind qualifier
// CHECK-ERROR:        sqdmlal2 v0.4s, v1.8s, v2.8h
// CHECK-ERROR: [[@LINE-4]]:25: error: invalid vector kind qualifier
// CHECK-ERROR:        sqdmlal2 v0.2d, v1.4d, v2.4s

        // Mismatched vector types
        sqdmlal v0.8h, v1.8b, v2.8b
        sqdmlal2 v0.8h, v1.16b, v2.16b

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlal v0.8h, v1.8b, v2.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlal2 v0.8h, v1.16b, v2.16b
// CHECK-ERROR:                    ^

        sqdmlsl v0.4s, v1.4s, v2.4h
        sqdmlsl v0.2d, v1.2d, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlsl v0.4s, v1.4s, v2.4h
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlsl v0.2d, v1.2d, v2.2s
// CHECK-ERROR:                          ^

        sqdmlsl2 v0.4s, v1.8s, v2.8h
        sqdmlsl2 v0.2d, v1.4d, v2.4s

// CHECK-ERROR: [[@LINE-3]]:25: error: invalid vector kind qualifier
// CHECK-ERROR:        sqdmlsl2 v0.4s, v1.8s, v2.8h
// CHECK-ERROR: [[@LINE-4]]:25: error: invalid vector kind qualifier
// CHECK-ERROR:        sqdmlsl2 v0.2d, v1.4d, v2.4s

        // Mismatched vector types
        sqdmlsl v0.8h, v1.8b, v2.8b
        sqdmlsl2 v0.8h, v1.16b, v2.16b

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlsl v0.8h, v1.8b, v2.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlsl2 v0.8h, v1.16b, v2.16b
// CHECK-ERROR:                    ^


        sqdmull v0.4s, v1.4s, v2.4h
        sqdmull v0.2d, v1.2d, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmull v0.4s, v1.4s, v2.4h
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmull v0.2d, v1.2d, v2.2s
// CHECK-ERROR:                          ^

        sqdmull2 v0.4s, v1.8s, v2.8h
        sqdmull2 v0.2d, v1.4d, v2.4s

// CHECK-ERROR: [[@LINE-3]]:25: error: invalid vector kind qualifier
// CHECK-ERROR:        sqdmull2 v0.4s, v1.8s, v2.8h
// CHECK-ERROR: [[@LINE-4]]:25: error: invalid vector kind qualifier
// CHECK-ERROR:        sqdmull2 v0.2d, v1.4d, v2.4s

        // Mismatched vector types
        sqdmull v0.8h, v1.8b, v2.8b
        sqdmull2 v0.8h, v1.16b, v2.16b

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmull v0.8h, v1.8b, v2.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmull2 v0.8h, v1.16b, v2.16b
// CHECK-ERROR:                    ^


//------------------------------------------------------------------------------
// Long - Variant 3
//------------------------------------------------------------------------------

        pmull v0.8h, v1.8h, v2.8b

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        pmull v0.8h, v1.8h, v2.8b
// CHECK-ERROR:                        ^

        pmull v0.1q, v1.2d, v2.2d
        
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        pmull v0.1q, v1.2d, v2.2d
// CHECK-ERROR:                     ^

        // Mismatched vector types
        pmull v0.4s, v1.4h, v2.4h
        pmull v0.2d, v1.2s, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        pmull v0.4s, v1.4h, v2.4h
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        pmull v0.2d, v1.2s, v2.2s
// CHECK-ERROR:                 ^


        pmull2 v0.8h, v1.16h, v2.16b
// CHECK-ERROR: [[@LINE-1]]:23: error: invalid vector kind qualifier
// CHECK-ERROR:        pmull2 v0.8h, v1.16h, v2.16b

        pmull2 v0.q, v1.2d, v2.2d
// CHECK-ERROR: [[@LINE-1]]:16: error: invalid vector kind qualifier
// CHECK-ERROR:        pmull2 v0.q, v1.2d, v2.2d

        // Mismatched vector types
        pmull2 v0.4s, v1.8h v2.8h
        pmull2 v0.2d, v1.4s, v2.4s


// CHECK-ERROR: error: unexpected token in argument list
// CHECK-ERROR:        pmull2 v0.4s, v1.8h v2.8h
// CHECK-ERROR:                            ^

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        pmull2 v0.2d, v1.4s, v2.4s
// CHECK-ERROR:                  ^

//------------------------------------------------------------------------------
// Widen
//------------------------------------------------------------------------------

        saddw v0.8h, v1.8h, v2.8h
        saddw v0.4s, v1.4s, v2.4s
        saddw v0.2d, v1.2d, v2.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        saddw v0.8h, v1.8h, v2.8h
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        saddw v0.4s, v1.4s, v2.4s
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        saddw v0.2d, v1.2d, v2.2d
// CHECK-ERROR:                               ^

        saddw2 v0.8h, v1.8h, v2.16h
        saddw2 v0.4s, v1.4s, v2.8s
        saddw2 v0.2d, v1.2d, v2.4d

// CHECK-ERROR: [[@LINE-4]]:30: error: invalid vector kind qualifier
// CHECK-ERROR:        saddw2 v0.8h, v1.8h, v2.16h
// CHECK-ERROR: [[@LINE-5]]:30: error: invalid vector kind qualifier
// CHECK-ERROR:        saddw2 v0.4s, v1.4s, v2.8s
// CHECK-ERROR: [[@LINE-6]]:30: error: invalid vector kind qualifier
// CHECK-ERROR:        saddw2 v0.2d, v1.2d, v2.4d

        uaddw v0.8h, v1.8h, v2.8h
        uaddw v0.4s, v1.4s, v2.4s
        uaddw v0.2d, v1.2d, v2.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uaddw v0.8h, v1.8h, v2.8h
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uaddw v0.4s, v1.4s, v2.4s
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uaddw v0.2d, v1.2d, v2.2d
// CHECK-ERROR:                               ^

        uaddw2 v0.8h, v1.8h, v2.16h
        uaddw2 v0.4s, v1.4s, v2.8s
        uaddw2 v0.2d, v1.2d, v2.4d

// CHECK-ERROR: [[@LINE-4]]:30: error: invalid vector kind qualifier
// CHECK-ERROR:        uaddw2 v0.8h, v1.8h, v2.16h
// CHECK-ERROR: [[@LINE-5]]:30: error: invalid vector kind qualifier
// CHECK-ERROR:        uaddw2 v0.4s, v1.4s, v2.8s
// CHECK-ERROR: [[@LINE-6]]:30: error: invalid vector kind qualifier
// CHECK-ERROR:        uaddw2 v0.2d, v1.2d, v2.4d

        ssubw v0.8h, v1.8h, v2.8h
        ssubw v0.4s, v1.4s, v2.4s
        ssubw v0.2d, v1.2d, v2.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        ssubw v0.8h, v1.8h, v2.8h
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        ssubw v0.4s, v1.4s, v2.4s
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        ssubw v0.2d, v1.2d, v2.2d
// CHECK-ERROR:                               ^

        ssubw2 v0.8h, v1.8h, v2.16h
        ssubw2 v0.4s, v1.4s, v2.8s
        ssubw2 v0.2d, v1.2d, v2.4d

// CHECK-ERROR: [[@LINE-4]]:30: error: invalid vector kind qualifier
// CHECK-ERROR:        ssubw2 v0.8h, v1.8h, v2.16h
// CHECK-ERROR: [[@LINE-5]]:30: error: invalid vector kind qualifier
// CHECK-ERROR:        ssubw2 v0.4s, v1.4s, v2.8s
// CHECK-ERROR: [[@LINE-6]]:30: error: invalid vector kind qualifier
// CHECK-ERROR:        ssubw2 v0.2d, v1.2d, v2.4d

        usubw v0.8h, v1.8h, v2.8h
        usubw v0.4s, v1.4s, v2.4s
        usubw v0.2d, v1.2d, v2.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        usubw v0.8h, v1.8h, v2.8h
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        usubw v0.4s, v1.4s, v2.4s
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        usubw v0.2d, v1.2d, v2.2d
// CHECK-ERROR:                               ^

        usubw2 v0.8h, v1.8h, v2.16h
        usubw2 v0.4s, v1.4s, v2.8s
        usubw2 v0.2d, v1.2d, v2.4d

// CHECK-ERROR: [[@LINE-4]]:30: error: invalid vector kind qualifier
// CHECK-ERROR:        usubw2 v0.8h, v1.8h, v2.16h
// CHECK-ERROR: [[@LINE-5]]:30: error: invalid vector kind qualifier
// CHECK-ERROR:        usubw2 v0.4s, v1.4s, v2.8s
// CHECK-ERROR: [[@LINE-6]]:30: error: invalid vector kind qualifier
// CHECK-ERROR:        usubw2 v0.2d, v1.2d, v2.4d

//------------------------------------------------------------------------------
// Narrow
//------------------------------------------------------------------------------

        addhn v0.8b, v1.8h, v2.8d
        addhn v0.4h, v1.4s, v2.4h
        addhn v0.2s, v1.2d, v2.2s

// CHECK-ERROR: [[@LINE-4]]:29: error: invalid vector kind qualifier
// CHECK-ERROR:        addhn v0.8b, v1.8h, v2.8d
// CHECK-ERROR: [[@LINE-5]]:29: error: invalid operand for instruction
// CHECK-ERROR:        addhn v0.4h, v1.4s, v2.4h
// CHECK-ERROR: [[@LINE-6]]:29: error: invalid operand for instruction
// CHECK-ERROR:        addhn v0.2s, v1.2d, v2.2s

        addhn2 v0.16b, v1.8h, v2.8b
        addhn2 v0.8h, v1.4s, v2.4h
        addhn2 v0.4s, v1.2d, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        addhn2 v0.16b, v1.8h, v2.8b
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        addhn2 v0.8h, v1.4s, v2.4h
// CHECK-ERROR:                                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        addhn2 v0.4s, v1.2d, v2.2s
// CHECK-ERROR:                                ^

        raddhn v0.8b, v1.8h, v2.8b
        raddhn v0.4h, v1.4s, v2.4h
        raddhn v0.2s, v1.2d, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        raddhn v0.8b, v1.8h, v2.8b
// CHECK-ERROR:                                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        raddhn v0.4h, v1.4s, v2.4h
// CHECK-ERROR:                                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        raddhn v0.2s, v1.2d, v2.2s
// CHECK-ERROR:                                ^

        raddhn2 v0.16b, v1.8h, v2.8b
        raddhn2 v0.8h, v1.4s, v2.4h
        raddhn2 v0.4s, v1.2d, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        raddhn2 v0.16b, v1.8h, v2.8b
// CHECK-ERROR:                                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        raddhn2 v0.8h, v1.4s, v2.4h
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        raddhn2 v0.4s, v1.2d, v2.2s
// CHECK-ERROR:                                 ^

        rsubhn v0.8b, v1.8h, v2.8b
        rsubhn v0.4h, v1.4s, v2.4h
        rsubhn v0.2s, v1.2d, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        rsubhn v0.8b, v1.8h, v2.8b
// CHECK-ERROR:                                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        rsubhn v0.4h, v1.4s, v2.4h
// CHECK-ERROR:                                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        rsubhn v0.2s, v1.2d, v2.2s
// CHECK-ERROR:                                ^

        rsubhn2 v0.16b, v1.8h, v2.8b
        rsubhn2 v0.8h, v1.4s, v2.4h
        rsubhn2 v0.4s, v1.2d, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        rsubhn2 v0.16b, v1.8h, v2.8b
// CHECK-ERROR:                                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        rsubhn2 v0.8h, v1.4s, v2.4h
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        rsubhn2 v0.4s, v1.2d, v2.2s
// CHECK-ERROR:                                 ^

//----------------------------------------------------------------------
// Scalar Reduce Add Pairwise (Integer)
//----------------------------------------------------------------------
         // invalid vector types
      addp s0, d1.2d
      addp d0, d1.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          addp s0, d1.2d
// CHECK-ERROR:               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          addp d0, d1.2s
// CHECK-ERROR:                      ^

//----------------------------------------------------------------------
// Scalar Reduce Add Pairwise (Floating Point)
//----------------------------------------------------------------------
         // invalid vector types
      faddp s0, d1.2d
      faddp d0, d1.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          faddp s0, d1.2d
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          faddp d0, d1.2s
// CHECK-ERROR:                    ^

//----------------------------------------------------------------------
// Scalar Reduce Maximum Pairwise (Floating Point)
//----------------------------------------------------------------------
         // mismatched and invalid vector types
      fmaxp s0, v1.2d
      fmaxp d31, v2.2s
      fmaxp h3, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fmaxp s0, v1.2d
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fmaxp d31, v2.2s
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fmaxp h3, v2.2s
// CHECK-ERROR:                ^


//----------------------------------------------------------------------
// Scalar Reduce Minimum Pairwise (Floating Point)
//----------------------------------------------------------------------
         // mismatched and invalid vector types
      fminp s0, v1.4h
      fminp d31, v2.8h
      fminp b3, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fminp s0, v1.4h
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fminp d31, v2.8h
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fminp b3, v2.2s
// CHECK-ERROR:                ^


//----------------------------------------------------------------------
// Scalar Reduce maxNum Pairwise (Floating Point)
//----------------------------------------------------------------------
         // mismatched and invalid vector types
      fmaxnmp s0, v1.8b
      fmaxnmp d31, v2.16b
      fmaxnmp v1.2s, v2.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fmaxnmp s0, v1.8b
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fmaxnmp d31, v2.16b
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: too few operands for instruction
// CHECK-ERROR:          fmaxnmp v1.2s, v2.2s
// CHECK-ERROR:          ^

//----------------------------------------------------------------------
// Scalar Reduce minNum Pairwise (Floating Point)
//----------------------------------------------------------------------
         // mismatched and invalid vector types
      fminnmp s0, v1.2d
      fminnmp d31, v2.4s
      fminnmp v1.4s, v2.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fminnmp s0, v1.2d
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fminnmp d31, v2.4s
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fminnmp v1.4s, v2.2d
// CHECK-ERROR:          ^

      mla v0.2d, v1.2d, v16.d[1]
      mla v0.2s, v1.2s, v2.s[4]
      mla v0.4s, v1.4s, v2.s[4]
      mla v0.2h, v1.2h, v2.h[1]
      mla v0.4h, v1.4h, v2.h[8]
      mla v0.8h, v1.8h, v2.h[8]
      mla v0.4h, v1.4h, v16.h[2]
      mla v0.8h, v1.8h, v16.h[2]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        mla v0.2d, v1.2d, v16.d[1]
// CHECK-ERROR:               ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        mla v0.2s, v1.2s, v2.s[4]
// CHECK-ERROR:                               ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        mla v0.4s, v1.4s, v2.s[4]
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        mla v0.2h, v1.2h, v2.h[1]
// CHECK-ERROR:            ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        mla v0.4h, v1.4h, v2.h[8]
// CHECK-ERROR:                               ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        mla v0.8h, v1.8h, v2.h[8]
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        mla v0.4h, v1.4h, v16.h[2]
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        mla v0.8h, v1.8h, v16.h[2]
// CHECK-ERROR:                              ^

      mls v0.2d, v1.2d, v16.d[1]
      mls v0.2s, v1.2s, v2.s[4]
      mls v0.4s, v1.4s, v2.s[4]
      mls v0.2h, v1.2h, v2.h[1]
      mls v0.4h, v1.4h, v2.h[8]
      mls v0.8h, v1.8h, v2.h[8]
      mls v0.4h, v1.4h, v16.h[2]
      mls v0.8h, v1.8h, v16.h[2]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        mls v0.2d, v1.2d, v16.d[1]
// CHECK-ERROR:               ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        mls v0.2s, v1.2s, v2.s[4]
// CHECK-ERROR:                               ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        mls v0.4s, v1.4s, v2.s[4]
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        mls v0.2h, v1.2h, v2.h[1]
// CHECK-ERROR:            ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        mls v0.4h, v1.4h, v2.h[8]
// CHECK-ERROR:                               ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        mls v0.8h, v1.8h, v2.h[8]
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        mls v0.4h, v1.4h, v16.h[2]
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        mls v0.8h, v1.8h, v16.h[2]
// CHECK-ERROR:                              ^

      fmla v0.4h, v1.4h, v2.h[2]
      fmla v0.8h, v1.8h, v2.h[2]
      fmla v0.2s, v1.2s, v2.s[4]
      fmla v0.2s, v1.2s, v22.s[4]
      fmla v3.4s, v8.4s, v2.s[4]
      fmla v3.4s, v8.4s, v22.s[4]
      fmla v0.2d, v1.2d, v2.d[2]
      fmla v0.2d, v1.2d, v22.d[2]

// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:        fmla v0.4h, v1.4h, v2.h[2]
// CHECK-ERROR:                ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:        fmla v0.8h, v1.8h, v2.h[2]
// CHECK-ERROR:                ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmla v0.2s, v1.2s, v2.s[4]
// CHECK-ERROR:                                ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmla v0.2s, v1.2s, v22.s[4]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmla v3.4s, v8.4s, v2.s[4]
// CHECK-ERROR:                                ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmla v3.4s, v8.4s, v22.s[4]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmla v0.2d, v1.2d, v2.d[2]
// CHECK-ERROR:                                ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmla v0.2d, v1.2d, v22.d[2]
// CHECK-ERROR:                                 ^

      fmls v0.4h, v1.4h, v2.h[2]
      fmls v0.8h, v1.8h, v2.h[2]
      fmls v0.2s, v1.2s, v2.s[4]
      fmls v0.2s, v1.2s, v22.s[4]
      fmls v3.4s, v8.4s, v2.s[4]
      fmls v3.4s, v8.4s, v22.s[4]
      fmls v0.2d, v1.2d, v2.d[2]
      fmls v0.2d, v1.2d, v22.d[2]

// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:        fmls v0.4h, v1.4h, v2.h[2]
// CHECK-ERROR:                ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:        fmls v0.8h, v1.8h, v2.h[2]
// CHECK-ERROR:                ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmls v0.2s, v1.2s, v2.s[4]
// CHECK-ERROR:                                ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmls v0.2s, v1.2s, v22.s[4]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmls v3.4s, v8.4s, v2.s[4]
// CHECK-ERROR:                                ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmls v3.4s, v8.4s, v22.s[4]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmls v0.2d, v1.2d, v2.d[2]
// CHECK-ERROR:                                ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmls v0.2d, v1.2d, v22.d[2]
// CHECK-ERROR:                                 ^

      smlal v0.4h, v1.4h, v2.h[2]
      smlal v0.4s, v1.4h, v2.h[8]
      smlal v0.4s, v1.4h, v16.h[2]
      smlal v0.2s, v1.2s, v2.s[1]
      smlal v0.2d, v1.2s, v2.s[4]
      smlal v0.2d, v1.2s, v22.s[4]
      smlal2 v0.4h, v1.8h, v1.h[2]
      smlal2 v0.4s, v1.8h, v1.h[8]
      smlal2 v0.4s, v1.8h, v16.h[2]
      smlal2 v0.2s, v1.4s, v1.s[2]
      smlal2 v0.2d, v1.4s, v1.s[4]
      smlal2 v0.2d, v1.4s, v22.s[4]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smlal v0.4h, v1.4h, v2.h[2]
// CHECK-ERROR:              ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        smlal v0.4s, v1.4h, v2.h[8]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smlal v0.4s, v1.4h, v16.h[2]
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smlal v0.2s, v1.2s, v2.s[1]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        smlal v0.2d, v1.2s, v2.s[4]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        smlal v0.2d, v1.2s, v22.s[4]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smlal2 v0.4h, v1.8h, v1.h[2]
// CHECK-ERROR:               ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        smlal2 v0.4s, v1.8h, v1.h[8]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smlal2 v0.4s, v1.8h, v16.h[2]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smlal2 v0.2s, v1.4s, v1.s[2]
// CHECK-ERROR:               ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        smlal2 v0.2d, v1.4s, v1.s[4]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        smlal2 v0.2d, v1.4s, v22.s[4]
// CHECK-ERROR:                                   ^

      smlsl v0.4h, v1.4h, v2.h[2]
      smlsl v0.4s, v1.4h, v2.h[8]
      smlsl v0.4s, v1.4h, v16.h[2]
      smlsl v0.2s, v1.2s, v2.s[1]
      smlsl v0.2d, v1.2s, v2.s[4]
      smlsl v0.2d, v1.2s, v22.s[4]
      smlsl2 v0.4h, v1.8h, v1.h[2]
      smlsl2 v0.4s, v1.8h, v1.h[8]
      smlsl2 v0.4s, v1.8h, v16.h[2]
      smlsl2 v0.2s, v1.4s, v1.s[2]
      smlsl2 v0.2d, v1.4s, v1.s[4]
      smlsl2 v0.2d, v1.4s, v22.s[4]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smlsl v0.4h, v1.4h, v2.h[2]
// CHECK-ERROR:              ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        smlsl v0.4s, v1.4h, v2.h[8]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smlsl v0.4s, v1.4h, v16.h[2]
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smlsl v0.2s, v1.2s, v2.s[1]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        smlsl v0.2d, v1.2s, v2.s[4]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        smlsl v0.2d, v1.2s, v22.s[4]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smlsl2 v0.4h, v1.8h, v1.h[2]
// CHECK-ERROR:               ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        smlsl2 v0.4s, v1.8h, v1.h[8]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smlsl2 v0.4s, v1.8h, v16.h[2]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smlsl2 v0.2s, v1.4s, v1.s[2]
// CHECK-ERROR:               ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        smlsl2 v0.2d, v1.4s, v1.s[4]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        smlsl2 v0.2d, v1.4s, v22.s[4]
// CHECK-ERROR:                                   ^

      umlal v0.4h, v1.4h, v2.h[2]
      umlal v0.4s, v1.4h, v2.h[8]
      umlal v0.4s, v1.4h, v16.h[2]
      umlal v0.2s, v1.2s, v2.s[1]
      umlal v0.2d, v1.2s, v2.s[4]
      umlal v0.2d, v1.2s, v22.s[4]
      umlal2 v0.4h, v1.8h, v1.h[2]
      umlal2 v0.4s, v1.8h, v1.h[8]
      umlal2 v0.4s, v1.8h, v16.h[2]
      umlal2 v0.2s, v1.4s, v1.s[2]
      umlal2 v0.2d, v1.4s, v1.s[4]
      umlal2 v0.2d, v1.4s, v22.s[4]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umlal v0.4h, v1.4h, v2.h[2]
// CHECK-ERROR:              ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        umlal v0.4s, v1.4h, v2.h[8]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umlal v0.4s, v1.4h, v16.h[2]
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umlal v0.2s, v1.2s, v2.s[1]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        umlal v0.2d, v1.2s, v2.s[4]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        umlal v0.2d, v1.2s, v22.s[4]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umlal2 v0.4h, v1.8h, v1.h[2]
// CHECK-ERROR:               ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        umlal2 v0.4s, v1.8h, v1.h[8]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umlal2 v0.4s, v1.8h, v16.h[2]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umlal2 v0.2s, v1.4s, v1.s[2]
// CHECK-ERROR:               ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        umlal2 v0.2d, v1.4s, v1.s[4]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        umlal2 v0.2d, v1.4s, v22.s[4]
// CHECK-ERROR:                                   ^

      umlsl v0.4h, v1.4h, v2.h[2]
      umlsl v0.4s, v1.4h, v2.h[8]
      umlsl v0.4s, v1.4h, v16.h[2]
      umlsl v0.2s, v1.2s, v2.s[3]
      umlsl v0.2d, v1.2s, v2.s[4]
      umlsl v0.2d, v1.2s, v22.s[4]
      umlsl2 v0.4h, v1.8h, v1.h[2]
      umlsl2 v0.4s, v1.8h, v1.h[8]
      umlsl2 v0.4s, v1.8h, v16.h[2]
      umlsl2 v0.2s, v1.4s, v1.s[2]
      umlsl2 v0.2d, v1.4s, v1.s[4]
      umlsl2 v0.2d, v1.4s, v22.s[4]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umlsl v0.4h, v1.4h, v2.h[2]
// CHECK-ERROR:              ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        umlsl v0.4s, v1.4h, v2.h[8]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umlsl v0.4s, v1.4h, v16.h[2]
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umlsl v0.2s, v1.2s, v2.s[3]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        umlsl v0.2d, v1.2s, v2.s[4]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        umlsl v0.2d, v1.2s, v22.s[4]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umlsl2 v0.4h, v1.8h, v1.h[2]
// CHECK-ERROR:               ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        umlsl2 v0.4s, v1.8h, v1.h[8]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umlsl2 v0.4s, v1.8h, v16.h[2]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umlsl2 v0.2s, v1.4s, v1.s[2]
// CHECK-ERROR:               ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        umlsl2 v0.2d, v1.4s, v1.s[4]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        umlsl2 v0.2d, v1.4s, v22.s[4]
// CHECK-ERROR:                                   ^

      sqdmlal v0.4h, v1.4h, v2.h[2]
      sqdmlal v0.4s, v1.4h, v2.h[8]
      sqdmlal v0.4s, v1.4h, v16.h[2]
      sqdmlal v0.2s, v1.2s, v2.s[3]
      sqdmlal v0.2d, v1.2s, v2.s[4]
      sqdmlal v0.2d, v1.2s, v22.s[4]
      sqdmlal2 v0.4h, v1.8h, v1.h[2]
      sqdmlal2 v0.4s, v1.8h, v1.h[8]
      sqdmlal2 v0.4s, v1.8h, v16.h[2]
      sqdmlal2 v0.2s, v1.4s, v1.s[2]
      sqdmlal2 v0.2d, v1.4s, v1.s[4]
      sqdmlal2 v0.2d, v1.4s, v22.s[4]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlal v0.4h, v1.4h, v2.h[2]
// CHECK-ERROR:                ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmlal v0.4s, v1.4h, v2.h[8]
// CHECK-ERROR:                                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlal v0.4s, v1.4h, v16.h[2]
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlal v0.2s, v1.2s, v2.s[3]
// CHECK-ERROR:                                   ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmlal v0.2d, v1.2s, v2.s[4]
// CHECK-ERROR:                                   ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmlal v0.2d, v1.2s, v22.s[4]
// CHECK-ERROR:                                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlal2 v0.4h, v1.8h, v1.h[2]
// CHECK-ERROR:                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmlal2 v0.4s, v1.8h, v1.h[8]
// CHECK-ERROR:                                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlal2 v0.4s, v1.8h, v16.h[2]
// CHECK-ERROR:                                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlal2 v0.2s, v1.4s, v1.s[2]
// CHECK-ERROR:                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmlal2 v0.2d, v1.4s, v1.s[4]
// CHECK-ERROR:                                    ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmlal2 v0.2d, v1.4s, v22.s[4]
// CHECK-ERROR:                                     ^

      sqdmlsl v0.4h, v1.4h, v2.h[2]
      sqdmlsl v0.4s, v1.4h, v2.h[8]
      sqdmlsl v0.4s, v1.4h, v16.h[2]
      sqdmlsl v0.2s, v1.2s, v2.s[3]
      sqdmlsl v0.2d, v1.2s, v2.s[4]
      sqdmlsl v0.2d, v1.2s, v22.s[4]
      sqdmlsl2 v0.4h, v1.8h, v1.h[2]
      sqdmlsl2 v0.4s, v1.8h, v1.h[8]
      sqdmlsl2 v0.4s, v1.8h, v16.h[2]
      sqdmlsl2 v0.2s, v1.4s, v1.s[2]
      sqdmlsl2 v0.2d, v1.4s, v1.s[4]
      sqdmlsl2 v0.2d, v1.4s, v22.s[4]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlsl v0.4h, v1.4h, v2.h[2]
// CHECK-ERROR:                ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmlsl v0.4s, v1.4h, v2.h[8]
// CHECK-ERROR:                                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlsl v0.4s, v1.4h, v16.h[2]
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlsl v0.2s, v1.2s, v2.s[3]
// CHECK-ERROR:                                   ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmlsl v0.2d, v1.2s, v2.s[4]
// CHECK-ERROR:                                   ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmlsl v0.2d, v1.2s, v22.s[4]
// CHECK-ERROR:                                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlsl2 v0.4h, v1.8h, v1.h[2]
// CHECK-ERROR:                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmlsl2 v0.4s, v1.8h, v1.h[8]
// CHECK-ERROR:                                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlsl2 v0.4s, v1.8h, v16.h[2]
// CHECK-ERROR:                                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlsl2 v0.2s, v1.4s, v1.s[2]
// CHECK-ERROR:                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmlsl2 v0.2d, v1.4s, v1.s[4]
// CHECK-ERROR:                                    ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmlsl2 v0.2d, v1.4s, v22.s[4]
// CHECK-ERROR:                                     ^

      mul v0.4h, v1.4h, v2.h[8]
      mul v0.4h, v1.4h, v16.h[8]
      mul v0.8h, v1.8h, v2.h[8]
      mul v0.8h, v1.8h, v16.h[8]
      mul v0.2s, v1.2s, v2.s[4]
      mul v0.2s, v1.2s, v22.s[4]
      mul v0.4s, v1.4s, v2.s[4]
      mul v0.4s, v1.4s, v22.s[4]
      mul v0.2d, v1.2d, v2.d[1]

// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        mul v0.4h, v1.4h, v2.h[8]
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        mul v0.4h, v1.4h, v16.h[8]
// CHECK-ERROR:                                ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        mul v0.8h, v1.8h, v2.h[8]
// CHECK-ERROR:                               ^
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR:        mul v0.8h, v1.8h, v16.h[8]
// CHECK-ERROR:                                ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        mul v0.2s, v1.2s, v2.s[4]
// CHECK-ERROR:                               ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        mul v0.2s, v1.2s, v22.s[4]
// CHECK-ERROR:                                ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        mul v0.4s, v1.4s, v2.s[4]
// CHECK-ERROR:                               ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        mul v0.4s, v1.4s, v22.s[4]
// CHECK-ERROR:                                ^

      fmul v0.4h, v1.4h, v2.h[4]
      fmul v0.2s, v1.2s, v2.s[4]
      fmul v0.2s, v1.2s, v22.s[4]
      fmul v0.4s, v1.4s, v2.s[4]
      fmul v0.4s, v1.4s, v22.s[4]
      fmul v0.2d, v1.2d, v2.d[2]
      fmul v0.2d, v1.2d, v22.d[2]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        mul v0.2d, v1.2d, v2.d[1]
// CHECK-ERROR:               ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:        fmul v0.4h, v1.4h, v2.h[4]
// CHECK-ERROR:                ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmul v0.2s, v1.2s, v2.s[4]
// CHECK-ERROR:                                ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmul v0.2s, v1.2s, v22.s[4]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmul v0.4s, v1.4s, v2.s[4]
// CHECK-ERROR:                                ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmul v0.4s, v1.4s, v22.s[4]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmul v0.2d, v1.2d, v2.d[2]
// CHECK-ERROR:                                ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmul v0.2d, v1.2d, v22.d[2]
// CHECK-ERROR:                                 ^

      fmulx v0.4h, v1.4h, v2.h[4]
      fmulx v0.2s, v1.2s, v2.s[4]
      fmulx v0.2s, v1.2s, v22.s[4]
      fmulx v0.4s, v1.4s, v2.s[4]
      fmulx v0.4s, v1.4s, v22.s[4]
      fmulx v0.2d, v1.2d, v2.d[2]
      fmulx v0.2d, v1.2d, v22.d[2]

// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:        fmulx v0.4h, v1.4h, v2.h[4]
// CHECK-ERROR:                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmulx v0.2s, v1.2s, v2.s[4]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmulx v0.2s, v1.2s, v22.s[4]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmulx v0.4s, v1.4s, v2.s[4]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmulx v0.4s, v1.4s, v22.s[4]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmulx v0.2d, v1.2d, v2.d[2]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        fmulx v0.2d, v1.2d, v22.d[2]
// CHECK-ERROR:                                  ^

      smull v0.4h, v1.4h, v2.h[2]
      smull v0.4s, v1.4h, v2.h[8]
      smull v0.4s, v1.4h, v16.h[4]
      smull v0.2s, v1.2s, v2.s[2]
      smull v0.2d, v1.2s, v2.s[4]
      smull v0.2d, v1.2s, v22.s[4]
      smull2 v0.4h, v1.8h, v2.h[2]
      smull2 v0.4s, v1.8h, v2.h[8]
      smull2 v0.4s, v1.8h, v16.h[4]
      smull2 v0.2s, v1.4s, v2.s[2]
      smull2 v0.2d, v1.4s, v2.s[4]
      smull2 v0.2d, v1.4s, v22.s[4]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smull v0.4h, v1.4h, v2.h[2]
// CHECK-ERROR:              ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        smull v0.4s, v1.4h, v2.h[8]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smull v0.4s, v1.4h, v16.h[4]
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smull v0.2s, v1.2s, v2.s[2]
// CHECK-ERROR:              ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        smull v0.2d, v1.2s, v2.s[4]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        smull v0.2d, v1.2s, v22.s[4]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smull2 v0.4h, v1.8h, v2.h[2]
// CHECK-ERROR:               ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        smull2 v0.4s, v1.8h, v2.h[8]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smull2 v0.4s, v1.8h, v16.h[4]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smull2 v0.2s, v1.4s, v2.s[2]
// CHECK-ERROR:               ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        smull2 v0.2d, v1.4s, v2.s[4]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        smull2 v0.2d, v1.4s, v22.s[4]
// CHECK-ERROR:                                   ^

      umull v0.4h, v1.4h, v2.h[2]
      umull v0.4s, v1.4h, v2.h[8]
      umull v0.4s, v1.4h, v16.h[4]
      umull v0.2s, v1.2s, v2.s[2]
      umull v0.2d, v1.2s, v2.s[4]
      umull v0.2d, v1.2s, v22.s[4]
      umull2 v0.4h, v1.8h, v2.h[2]
      umull2 v0.4s, v1.8h, v2.h[8]
      umull2 v0.4s, v1.8h, v16.h[4]
      umull2 v0.2s, v1.4s, v2.s[2]
      umull2 v0.2d, v1.4s, v2.s[4]
      umull2 v0.2d, v1.4s, v22.s[4]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umull v0.4h, v1.4h, v2.h[2]
// CHECK-ERROR:              ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        umull v0.4s, v1.4h, v2.h[8]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umull v0.4s, v1.4h, v16.h[4]
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umull v0.2s, v1.2s, v2.s[2]
// CHECK-ERROR:              ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        umull v0.2d, v1.2s, v2.s[4]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        umull v0.2d, v1.2s, v22.s[4]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umull2 v0.4h, v1.8h, v2.h[2]
// CHECK-ERROR:               ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        umull2 v0.4s, v1.8h, v2.h[8]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umull2 v0.4s, v1.8h, v16.h[4]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umull2 v0.2s, v1.4s, v2.s[2]
// CHECK-ERROR:               ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        umull2 v0.2d, v1.4s, v2.s[4]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        umull2 v0.2d, v1.4s, v22.s[4]
// CHECK-ERROR:                                   ^

      sqdmull v0.4h, v1.4h, v2.h[2]
      sqdmull v0.4s, v1.4h, v2.h[8]
      sqdmull v0.4s, v1.4h, v16.h[4]
      sqdmull v0.2s, v1.2s, v2.s[2]
      sqdmull v0.2d, v1.2s, v2.s[4]
      sqdmull v0.2d, v1.2s, v22.s[4]
      sqdmull2 v0.4h, v1.8h, v2.h[2]
      sqdmull2 v0.4s, v1.8h, v2.h[8]
      sqdmull2 v0.4s, v1.8h, v16.h[4]
      sqdmull2 v0.2s, v1.4s, v2.s[2]
      sqdmull2 v0.2d, v1.4s, v2.s[4]
      sqdmull2 v0.2d, v1.4s, v22.s[4]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmull v0.4h, v1.4h, v2.h[2]
// CHECK-ERROR:                ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmull v0.4s, v1.4h, v2.h[8]
// CHECK-ERROR:                                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmull v0.4s, v1.4h, v16.h[4]
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmull v0.2s, v1.2s, v2.s[2]
// CHECK-ERROR:                ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmull v0.2d, v1.2s, v2.s[4]
// CHECK-ERROR:                                   ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmull v0.2d, v1.2s, v22.s[4]
// CHECK-ERROR:                                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmull2 v0.4h, v1.8h, v2.h[2]
// CHECK-ERROR:                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmull2 v0.4s, v1.8h, v2.h[8]
// CHECK-ERROR:                                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmull2 v0.4s, v1.8h, v16.h[4]
// CHECK-ERROR:                                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmull2 v0.2s, v1.4s, v2.s[2]
// CHECK-ERROR:                 ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmull2 v0.2d, v1.4s, v2.s[4]
// CHECK-ERROR:                                    ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmull2 v0.2d, v1.4s, v22.s[4]
// CHECK-ERROR:                                     ^

      sqdmulh v0.4h, v1.4h, v2.h[8]
      sqdmulh v0.4h, v1.4h, v16.h[2]
      sqdmulh v0.8h, v1.8h, v2.h[8]
      sqdmulh v0.8h, v1.8h, v16.h[2]
      sqdmulh v0.2s, v1.2s, v2.s[4]
      sqdmulh v0.2s, v1.2s, v22.s[4]
      sqdmulh v0.4s, v1.4s, v2.s[4]
      sqdmulh v0.4s, v1.4s, v22.s[4]
      sqdmulh v0.2d, v1.2d, v22.d[1]

// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmulh v0.4h, v1.4h, v2.h[8]
// CHECK-ERROR:                                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmulh v0.4h, v1.4h, v16.h[2]
// CHECK-ERROR:                              ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmulh v0.8h, v1.8h, v2.h[8]
// CHECK-ERROR:                                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmulh v0.8h, v1.8h, v16.h[2]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmulh v0.2s, v1.2s, v2.s[4]
// CHECK-ERROR:                                   ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmulh v0.2s, v1.2s, v22.s[4]
// CHECK-ERROR:                                    ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmulh v0.4s, v1.4s, v2.s[4]
// CHECK-ERROR:                                   ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqdmulh v0.4s, v1.4s, v22.s[4]
// CHECK-ERROR:                                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmulh v0.2d, v1.2d, v22.d[1]
// CHECK-ERROR:                   ^

      sqrdmulh v0.4h, v1.4h, v2.h[8]
      sqrdmulh v0.4h, v1.4h, v16.h[2]
      sqrdmulh v0.8h, v1.8h, v2.h[8]
      sqrdmulh v0.8h, v1.8h, v16.h[2]
      sqrdmulh v0.2s, v1.2s, v2.s[4]
      sqrdmulh v0.2s, v1.2s, v22.s[4]
      sqrdmulh v0.4s, v1.4s, v2.s[4]
      sqrdmulh v0.4s, v1.4s, v22.s[4]
      sqrdmulh v0.2d, v1.2d, v22.d[1]

// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqrdmulh v0.4h, v1.4h, v2.h[8]
// CHECK-ERROR:                                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqrdmulh v0.4h, v1.4h, v16.h[2]
// CHECK-ERROR:                               ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqrdmulh v0.8h, v1.8h, v2.h[8]
// CHECK-ERROR:                                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqrdmulh v0.8h, v1.8h, v16.h[2]
// CHECK-ERROR:                                   ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqrdmulh v0.2s, v1.2s, v2.s[4]
// CHECK-ERROR:                                    ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqrdmulh v0.2s, v1.2s, v22.s[4]
// CHECK-ERROR:                                     ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqrdmulh v0.4s, v1.4s, v2.s[4]
// CHECK-ERROR:                                    ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:        sqrdmulh v0.4s, v1.4s, v22.s[4]
// CHECK-ERROR:                                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqrdmulh v0.2d, v1.2d, v22.d[1]
// CHECK-ERROR:                    ^

//----------------------------------------------------------------------
// Across vectors
//----------------------------------------------------------------------

        saddlv b0, v1.8b
        saddlv b0, v1.16b
        saddlv h0, v1.4h
        saddlv h0, v1.8h
        saddlv s0, v1.2s
        saddlv s0, v1.4s
        saddlv d0, v1.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        saddlv b0, v1.8b
// CHECK-ERROR:               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        saddlv b0, v1.16b
// CHECK-ERROR:               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        saddlv h0, v1.4h
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        saddlv h0, v1.8h
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        saddlv s0, v1.2s
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        saddlv s0, v1.4s
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        saddlv d0, v1.2s
// CHECK-ERROR:                   ^

        uaddlv b0, v1.8b
        uaddlv b0, v1.16b
        uaddlv h0, v1.4h
        uaddlv h0, v1.8h
        uaddlv s0, v1.2s
        uaddlv s0, v1.4s
        uaddlv d0, v1.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uaddlv b0, v1.8b
// CHECK-ERROR:               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uaddlv b0, v1.16b
// CHECK-ERROR:               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uaddlv h0, v1.4h
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uaddlv h0, v1.8h
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uaddlv s0, v1.2s
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uaddlv s0, v1.4s
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uaddlv d0, v1.2s
// CHECK-ERROR:                   ^

        smaxv s0, v1.2s
        sminv s0, v1.2s
        umaxv s0, v1.2s
        uminv s0, v1.2s
        addv s0, v1.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smaxv s0, v1.2s
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sminv s0, v1.2s
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umaxv s0, v1.2s
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uminv s0, v1.2s
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        addv s0, v1.2s
// CHECK-ERROR:                 ^

        smaxv d0, v1.2d
        sminv d0, v1.2d
        umaxv d0, v1.2d
        uminv d0, v1.2d
        addv d0, v1.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        smaxv d0, v1.2d
// CHECK-ERROR:              ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sminv d0, v1.2d
// CHECK-ERROR:              ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        umaxv d0, v1.2d
// CHECK-ERROR:              ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uminv d0, v1.2d
// CHECK-ERROR:              ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        addv d0, v1.2d
// CHECK-ERROR:             ^

        fmaxnmv b0, v1.16b
        fminnmv b0, v1.16b
        fmaxv b0, v1.16b
        fminv b0, v1.16b

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fmaxnmv b0, v1.16b
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fminnmv b0, v1.16b
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fmaxv b0, v1.16b
// CHECK-ERROR:              ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fminv b0, v1.16b
// CHECK-ERROR:              ^

        fmaxnmv h0, v1.8h
        fminnmv h0, v1.8h
        fmaxv h0, v1.8h
        fminv h0, v1.8h

// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:        fmaxnmv h0, v1.8h
// CHECK-ERROR:                ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:        fminnmv h0, v1.8h
// CHECK-ERROR:                ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:        fmaxv h0, v1.8h
// CHECK-ERROR:              ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:        fminv h0, v1.8h
// CHECK-ERROR:              ^

        fmaxnmv d0, v1.2d
        fminnmv d0, v1.2d
        fmaxv d0, v1.2d
        fminv d0, v1.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fmaxnmv d0, v1.2d
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fminnmv d0, v1.2d
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fmaxv d0, v1.2d
// CHECK-ERROR:              ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fminv d0, v1.2d
// CHECK-ERROR:              ^

//----------------------------------------------------------------------
// Floating-point Multiply Extended
//----------------------------------------------------------------------

    fmulx s20, h22, s15
    fmulx d23, d11, s1

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fmulx s20, h22, s15
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fmulx d23, d11, s1
// CHECK-ERROR:                          ^

//----------------------------------------------------------------------
// Floating-point Reciprocal Step
//----------------------------------------------------------------------

    frecps s21, s16, h13
    frecps d22, s30, d21

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          frecps s21, s16, h13
// CHECK-ERROR:                           ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          frecps d22, s30, d21
// CHECK-ERROR:                      ^

//----------------------------------------------------------------------
// Floating-point Reciprocal Square Root Step
//----------------------------------------------------------------------

    frsqrts s21, h5, s12
    frsqrts d8, s22, d18

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          frsqrts s21, h5, s12
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          frsqrts d8, s22, d18
// CHECK-ERROR:                      ^

//----------------------------------------------------------------------
// Vector load/store multiple N-element structure (class SIMD lselem)
//----------------------------------------------------------------------
         ld1 {x3}, [x2]
         ld1 {v4}, [x0]
         ld1 {v32.16b}, [x0]
         ld1 {v15.8h}, [x32]
// CHECK-ERROR: error: vector register expected
// CHECK-ERROR:        ld1 {x3}, [x2]
// CHECK-ERROR:             ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        ld1 {v4}, [x0]
// CHECK-ERROR:             ^
// CHECK-ERROR: error: vector register expected
// CHECK-ERROR:        ld1 {v32.16b}, [x0]
// CHECK-ERROR:             ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        ld1 {v15.8h}, [x32]
// CHECK-ERROR:                       ^

         ld1 {v0.16b, v2.16b}, [x0]
         ld1 {v0.8h, v1.8h, v2.8h, v3.8h, v4.8h}, [x0]
         ld1 v0.8b, v1.8b}, [x0]
         ld1 {v0.8h-v4.8h}, [x0]
         ld1 {v1.8h-v1.8h}, [x0]
         ld1 {v15.8h-v17.4h}, [x15]
         ld1 {v0.8b-v2.8b, [x0]
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        ld1 {v0.16b, v2.16b}, [x0]
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid number of vectors
// CHECK-ERROR:        ld1 {v0.8h, v1.8h, v2.8h, v3.8h, v4.8h}, [x0]
// CHECK-ERROR:                                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        ld1 v0.8b, v1.8b}, [x0]
// CHECK-ERROR:            ^
// CHECK-ERROR: error: invalid number of vectors
// CHECK-ERROR:        ld1 {v0.8h-v4.8h}, [x0]
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid number of vectors
// CHECK-ERROR:        ld1 {v1.8h-v1.8h}, [x0]
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: mismatched register size suffix
// CHECK-ERROR:        ld1 {v15.8h-v17.4h}, [x15]
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: '}' expected
// CHECK-ERROR:        ld1 {v0.8b-v2.8b, [x0]
// CHECK-ERROR:                        ^

         ld2 {v15.8h, v16.4h}, [x15]
         ld2 {v0.8b, v2.8b}, [x0]
         ld2 {v15.4h, v16.4h, v17.4h}, [x32]
         ld2 {v15.8h-v16.4h}, [x15]
         ld2 {v0.2d-v2.2d}, [x0]
// CHECK-ERROR: error: mismatched register size suffix
// CHECK-ERROR:        ld2 {v15.8h, v16.4h}, [x15]
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        ld2 {v0.8b, v2.8b}, [x0]
// CHECK-ERROR:                    ^
// CHECK-ERROR:        ld2 {v15.4h, v16.4h, v17.4h}, [x32]
// CHECK-ERROR:            ^
// CHECK-ERROR: error: mismatched register size suffix
// CHECK-ERROR:        ld2 {v15.8h-v16.4h}, [x15]
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        ld2 {v0.2d-v2.2d}, [x0]
// CHECK-ERROR:            ^

         ld3 {v15.8h, v16.8h, v17.4h}, [x15]
         ld3 {v0.8b, v1,8b, v2.8b, v3.8b}, [x0]
         ld3 {v0.8b, v2.8b, v3.8b}, [x0]
         ld3 {v15.8h-v17.4h}, [x15]
         ld3 {v31.4s-v2.4s}, [sp]
// CHECK-ERROR: error: mismatched register size suffix
// CHECK-ERROR:        ld3 {v15.8h, v16.8h, v17.4h}, [x15]
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: mismatched register size suffix
// CHECK-ERROR:        ld3 {v0.8b, v1,8b, v2.8b, v3.8b}, [x0]
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: registers must have the same sequential stride
// CHECK-ERROR:        ld3 {v0.8b, v2.8b, v3.8b}, [x0]
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: mismatched register size suffix
// CHECK-ERROR:        ld3 {v15.8h-v17.4h}, [x15]
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        ld3 {v31.4s-v2.4s}, [sp]
// CHECK-ERROR:            ^

         ld4 {v15.8h, v16.8h, v17.4h, v18.8h}, [x15]
         ld4 {v0.8b, v2.8b, v3.8b, v4.8b}, [x0]
         ld4 {v15.4h, v16.4h, v17.4h, v18.4h, v19.4h}, [x31]
         ld4 {v15.8h-v18.4h}, [x15]
         ld4 {v31.2s-v1.2s}, [x31]
// CHECK-ERROR: error: mismatched register size suffix
// CHECK-ERROR:        ld4 {v15.8h, v16.8h, v17.4h, v18.8h}, [x15]
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: registers must have the same sequential stride
// CHECK-ERROR:        ld4 {v0.8b, v2.8b, v3.8b, v4.8b}, [x0]
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid number of vectors
// CHECK-ERROR:        ld4 {v15.4h, v16.4h, v17.4h, v18.4h, v19.4h}, [x31]
// CHECK-ERROR:                                             ^
// CHECK-ERROR: error: mismatched register size suffix
// CHECK-ERROR:        ld4 {v15.8h-v18.4h}, [x15]
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        ld4 {v31.2s-v1.2s}, [x31]
// CHECK-ERROR:            ^

         st1 {x3}, [x2]
         st1 {v4}, [x0]
         st1 {v32.16b}, [x0]
         st1 {v15.8h}, [x32]
// CHECK-ERROR: error: vector register expected
// CHECK-ERROR:        st1 {x3}, [x2]
// CHECK-ERROR:             ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        st1 {v4}, [x0]
// CHECK-ERROR:             ^
// CHECK-ERROR: error: vector register expected
// CHECK-ERROR:        st1 {v32.16b}, [x0]
// CHECK-ERROR:             ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        st1 {v15.8h}, [x32]
// CHECK-ERROR:                       ^

         st1 {v0.16b, v2.16b}, [x0]
         st1 {v0.8h, v1.8h, v2.8h, v3.8h, v4.8h}, [x0]
         st1 v0.8b, v1.8b}, [x0]
         st1 {v0.8h-v4.8h}, [x0]
         st1 {v1.8h-v1.8h}, [x0]
         st1 {v15.8h-v17.4h}, [x15]
         st1 {v0.8b-v2.8b, [x0]
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        st1 {v0.16b, v2.16b}, [x0]
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid number of vectors
// CHECK-ERROR:        st1 {v0.8h, v1.8h, v2.8h, v3.8h, v4.8h}, [x0]
// CHECK-ERROR:                                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        st1 v0.8b, v1.8b}, [x0]
// CHECK-ERROR:            ^
// CHECK-ERROR: error: invalid number of vectors
// CHECK-ERROR:        st1 {v0.8h-v4.8h}, [x0]
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid number of vectors
// CHECK-ERROR:        st1 {v1.8h-v1.8h}, [x0]
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: mismatched register size suffix
// CHECK-ERROR:        st1 {v15.8h-v17.4h}, [x15]
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: '}' expected
// CHECK-ERROR:        st1 {v0.8b-v2.8b, [x0]
// CHECK-ERROR:                        ^

         st2 {v15.8h, v16.4h}, [x15]
         st2 {v0.8b, v2.8b}, [x0]
         st2 {v15.4h, v16.4h, v17.4h}, [x30]
         st2 {v15.8h-v16.4h}, [x15]
         st2 {v0.2d-v2.2d}, [x0]
// CHECK-ERROR: error: mismatched register size suffix
// CHECK-ERROR:        st2 {v15.8h, v16.4h}, [x15]
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        st2 {v0.8b, v2.8b}, [x0]
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        st2 {v15.4h, v16.4h, v17.4h}, [x30]
// CHECK-ERROR:            ^
// CHECK-ERROR: error: mismatched register size suffix
// CHECK-ERROR:        st2 {v15.8h-v16.4h}, [x15]
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        st2 {v0.2d-v2.2d}, [x0]
// CHECK-ERROR:            ^

         st3 {v15.8h, v16.8h, v17.4h}, [x15]
         st3 {v0.8b, v1,8b, v2.8b, v3.8b}, [x0]
         st3 {v0.8b, v2.8b, v3.8b}, [x0]
         st3 {v15.8h-v17.4h}, [x15]
         st3 {v31.4s-v2.4s}, [sp]
// CHECK-ERROR: error: mismatched register size suffix
// CHECK-ERROR:        st3 {v15.8h, v16.8h, v17.4h}, [x15]
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: mismatched register size suffix
// CHECK-ERROR:        st3 {v0.8b, v1,8b, v2.8b, v3.8b}, [x0]
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: registers must have the same sequential stride
// CHECK-ERROR:        st3 {v0.8b, v2.8b, v3.8b}, [x0]
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: mismatched register size suffix
// CHECK-ERROR:        st3 {v15.8h-v17.4h}, [x15]
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        st3 {v31.4s-v2.4s}, [sp]
// CHECK-ERROR:            ^

         st4 {v15.8h, v16.8h, v17.4h, v18.8h}, [x15]
         st4 {v0.8b, v2.8b, v3.8b, v4.8b}, [x0]
         st4 {v15.4h, v16.4h, v17.4h, v18.4h, v19.4h}, [x31]
         st4 {v15.8h-v18.4h}, [x15]
         st4 {v31.2s-v1.2s}, [x31]
// CHECK-ERROR: error: mismatched register size suffix
// CHECK-ERROR:        st4 {v15.8h, v16.8h, v17.4h, v18.8h}, [x15]
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: registers must have the same sequential stride
// CHECK-ERROR:        st4 {v0.8b, v2.8b, v3.8b, v4.8b}, [x0]
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid number of vectors
// CHECK-ERROR:        st4 {v15.4h, v16.4h, v17.4h, v18.4h, v19.4h}, [x31]
// CHECK-ERROR:                                             ^
// CHECK-ERROR: error: mismatched register size suffix
// CHECK-ERROR:        st4 {v15.8h-v18.4h}, [x15]
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        st4 {v31.2s-v1.2s}, [x31]
// CHECK-ERROR:            ^

//----------------------------------------------------------------------
// Vector post-index load/store multiple N-element structure
// (class SIMD lselem-post)
//----------------------------------------------------------------------
         ld1 {v0.16b}, [x0], #8
         ld1 {v0.8h, v1.16h}, [x0], x1
         ld1 {v0.8b, v1.8b, v2.8b, v3.8b}, [x0], #24
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          ld1 {v0.16b}, [x0], #8
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: invalid vector kind qualifier
// CHECK-ERROR:          ld1 {v0.8h, v1.16h}, [x0], x1
// CHECK-ERROR:                      ^
// CHECK-ERROR:  error: invalid operand for instruction
// CHECK-ERROR:          ld1 {v0.8b, v1.8b, v2.8b, v3.8b}, [x0], #24
// CHECK-ERROR:                                                  ^

         ld2 {v0.16b, v1.16b}, [x0], #16
         ld3 {v5.2s, v6.2s, v7.2s}, [x1], #48
         ld4 {v31.2d, v0.2d, v1.2d, v2.1d}, [x3], x1
// CHECK-ERROR:  error: invalid operand for instruction
// CHECK-ERROR:          ld2 {v0.16b, v1.16b}, [x0], #16
// CHECK-ERROR:                                      ^
// CHECK-ERROR:  error: invalid operand for instruction
// CHECK-ERROR:          ld3 {v5.2s, v6.2s, v7.2s}, [x1], #48
// CHECK-ERROR:                                           ^
// CHECK-ERROR: error: mismatched register size suffix
// CHECK-ERROR:          ld4 {v31.2d, v0.2d, v1.2d, v2.1d}, [x3], x1
// CHECK-ERROR:                                     ^

         st1 {v0.16b}, [x0], #8
         st1 {v0.8h, v1.16h}, [x0], x1
         st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [x0], #24
// CHECK-ERROR:  error: invalid operand for instruction
// CHECK-ERROR:          st1 {v0.16b}, [x0], #8
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: invalid vector kind qualifier
// CHECK-ERROR:          st1 {v0.8h, v1.16h}, [x0], x1
// CHECK-ERROR:                      ^
// CHECK-ERROR:  error: invalid operand for instruction
// CHECK-ERROR:          st1 {v0.8b, v1.8b, v2.8b, v3.8b}, [x0], #24
                                                 ^

         st2 {v0.16b, v1.16b}, [x0], #16
         st3 {v5.2s, v6.2s, v7.2s}, [x1], #48
         st4 {v31.2d, v0.2d, v1.2d, v2.1d}, [x3], x1
// CHECK-ERROR:  error: invalid operand for instruction
// CHECK-ERROR:          st2 {v0.16b, v1.16b}, [x0], #16
// CHECK-ERROR:                                      ^
// CHECK-ERROR:  error: invalid operand for instruction
// CHECK-ERROR:          st3 {v5.2s, v6.2s, v7.2s}, [x1], #48
// CHECK-ERROR:                                           ^
// CHECK-ERROR: error: mismatched register size suffix
// CHECK-ERROR:          st4 {v31.2d, v0.2d, v1.2d, v2.1d}, [x3], x1
// CHECK-ERROR:                                     ^

//------------------------------------------------------------------------------
// Load single N-element structure to all lanes of N consecutive
// registers (N = 1,2,3,4)
//------------------------------------------------------------------------------
         ld1r {x1}, [x0]
         ld2r {v31.4s, v0.2s}, [sp]
         ld3r {v0.8b, v1.8b, v2.8b, v3.8b}, [x0]
         ld4r {v31.2s, v0.2s, v1.2d, v2.2s}, [sp]
// CHECK-ERROR: error: vector register expected
// CHECK-ERROR: ld1r {x1}, [x0]
// CHECK-ERROR:       ^
// CHECK-ERROR: error: mismatched register size suffix
// CHECK-ERROR: ld2r {v31.4s, v0.2s}, [sp]
// CHECK-ERROR:               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: ld3r {v0.8b, v1.8b, v2.8b, v3.8b}, [x0]
// CHECK-ERROR:      ^
// CHECK-ERROR: error: mismatched register size suffix
// CHECK-ERROR: ld4r {v31.2s, v0.2s, v1.2d, v2.2s}, [sp]
// CHECK-ERROR:                      ^

//------------------------------------------------------------------------------
// Load/Store single N-element structure to/from one lane of N consecutive
// registers (N = 1, 2,3,4)
//------------------------------------------------------------------------------
         ld1 {v0.b}[16], [x0]
         ld2 {v15.h, v16.h}[8], [x15]
         ld3 {v31.s, v0.s, v1.s}[-1], [sp]
         ld4 {v0.d, v1.d, v2.d, v3.d}[2], [x0]
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR: ld1 {v0.b}[16], [x0]
// CHECK-ERROR:            ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR: ld2 {v15.h, v16.h}[8], [x15]
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: vector lane must be an integer in range
// CHECK-ERROR: ld3 {v31.s, v0.s, v1.s}[-1], [sp]
// CHECK-ERROR:                         ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR: ld4 {v0.d, v1.d, v2.d, v3.d}[2], [x0]
// CHECK-ERROR:                              ^

         st1 {v0.d}[16], [x0]
         st2 {v31.s, v0.s}[3], [8]
         st3 {v15.h, v16.h, v17.h}[-1], [x15]
         st4 {v0.d, v1.d, v2.d, v3.d}[2], [x0]
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR: st1 {v0.d}[16], [x0]
// CHECK-ERROR:            ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: st2 {v31.s, v0.s}[3], [8]
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: vector lane must be an integer in range
// CHECK-ERROR: st3 {v15.h, v16.h, v17.h}[-1], [x15]
// CHECK-ERROR:                           ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR: st4 {v0.d, v1.d, v2.d, v3.d}[2], [x0]
// CHECK-ERROR:                              ^

//------------------------------------------------------------------------------
// Post-index of load single N-element structure to all lanes of N consecutive
// registers (N = 1,2,3,4)
//------------------------------------------------------------------------------
         ld1r {v15.8h}, [x15], #5
         ld2r {v0.2d, v1.2d}, [x0], #7
         ld3r {v15.4h, v16.4h, v17.4h}, [x15], #1
         ld4r {v31.1d, v0.1d, v1.1d, v2.1d}, [sp], sp
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: ld1r {v15.8h}, [x15], #5
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: ld2r {v0.2d, v1.2d}, [x0], #7
// CHECK-ERROR:                            ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: ld3r {v15.4h, v16.4h, v17.4h}, [x15], #1
// CHECK-ERROR:                                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: ld4r {v31.1d, v0.1d, v1.1d, v2.1d}, [sp], sp
// CHECK-ERROR:                                           ^

//------------------------------------------------------------------------------
// Post-index of Load/Store single N-element structure to/from one lane of N
// consecutive registers (N = 1, 2,3,4)
//------------------------------------------------------------------------------
         ld1 {v0.b}[0], [x0], #2
         ld2 {v15.h, v16.h}[0], [x15], #3
         ld3 {v31.s, v0.s, v1.d}[0], [sp], x9
         ld4 {v0.d, v1.d, v2.d, v3.d}[1], [x0], #24
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: ld1 {v0.b}[0], [x0], #2
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: ld2 {v15.h, v16.h}[0], [x15], #3
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: mismatched register size suffix
// CHECK-ERROR: ld3 {v31.s, v0.s, v1.d}[0], [sp], x9
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: ld4 {v0.d, v1.d, v2.d, v3.d}[1], [x0], #24
// CHECK-ERROR:                                        ^

         st1 {v0.d}[0], [x0], #7
         st2 {v31.s, v0.s}[0], [sp], #6
         st3 {v15.h, v16.h, v17.h}[0], [x15], #8
         st4 {v0.b, v1.b, v2.b, v3.b}[1], [x0], #1
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: st1 {v0.d}[0], [x0], #7
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: st2 {v31.s, v0.s}[0], [sp], #6
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: st3 {v15.h, v16.h, v17.h}[0], [x15], #8
// CHECK-ERROR:                                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR: st4 {v0.b, v1.b, v2.b, v3.b}[1], [x0], #1
// CHECK-ERROR:                                        ^


         ins v2.b[16], w1
         ins v7.h[8], w14
         ins v20.s[5], w30
         ins v1.d[2], x7
         ins v2.b[3], b1
         ins v7.h[2], h14
         ins v20.s[1], s30
         ins v1.d[0], d7

// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:         ins v2.b[16], w1
// CHECK-ERROR:                  ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:         ins v7.h[8], w14
// CHECK-ERROR:                  ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:         ins v20.s[5], w30
// CHECK-ERROR:                   ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:         ins v1.d[2], x7
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ins v2.b[3], b1
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ins v7.h[2], h14
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ins v20.s[1], s30
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ins v1.d[0], d7
// CHECK-ERROR:                      ^

         smov w1, v0.b[16]
         smov w14, v6.h[8]
         smov x1, v0.b[16]
         smov x14, v6.h[8]
         smov x20, v9.s[5]
         smov w1, v0.d[0]
         smov w14, v6.d[1]
         smov x1, v0.d[0]
         smov x14, v6.d[1]
         smov x20, v9.d[0]

// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:         smov w1, v0.b[16]
// CHECK-ERROR:                       ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:         smov w14, v6.h[8]
// CHECK-ERROR:                        ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:         smov x1, v0.b[16]
// CHECK-ERROR:                       ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:         smov x14, v6.h[8]
// CHECK-ERROR:                        ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:         smov x20, v9.s[5]
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         smov w1, v0.d[0]
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         smov w14, v6.d[1]
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         smov x1, v0.d[0]
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         smov x14, v6.d[1]
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         smov x20, v9.d[0]
// CHECK-ERROR:                      ^

         umov w1, v0.b[16]
         umov w14, v6.h[8]
         umov w20, v9.s[5]
         umov x7, v18.d[3]
         umov w1, v0.d[0]
         umov s20, v9.s[2]
         umov d7, v18.d[1]

// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:         umov w1, v0.b[16]
// CHECK-ERROR:                       ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:         umov w14, v6.h[8]
// CHECK-ERROR:                        ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:         umov w20, v9.s[5]
// CHECK-ERROR:                        ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:         umov x7, v18.d[3]
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         umov w1, v0.d[0]
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         umov s20, v9.s[2]
// CHECK-ERROR:              ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         umov d7, v18.d[1]
// CHECK-ERROR:              ^

         Ins v1.h[2], v3.b[6]
         Ins v6.h[7], v7.s[2]
         Ins v15.d[0], v22.s[2]
         Ins v0.d[0], v4.b[1]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         Ins v1.h[2], v3.b[6]
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         Ins v6.h[7], v7.s[2]
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         Ins v15.d[0], v22.s[2]
// CHECK-ERROR:                           ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         Ins v0.d[0], v4.b[1]
// CHECK-ERROR:                         ^

         dup v1.8h, v2.b[2]
         dup v11.4s, v7.h[7]
         dup v17.2d, v20.s[0]
         dup v1.16b, v2.h[2]
         dup v11.8h, v7.s[3]
         dup v17.4s, v20.d[0]
         dup v5.2d, v1.b[1]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         dup v1.8h, v2.b[2]
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         dup v11.4s, v7.h[7]
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         dup v17.2d, v20.s[0]
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         dup v1.16b, v2.h[2]
// CHECK-ERROR:                        ^
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR:         dup v11.8h, v7.s[3]
// CHECK-ERROR:                        ^
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR:         dup v17.4s, v20.d[0]
// CHECK-ERROR:                         ^
// CHECK-ERROR: invalid operand for instruction
// CHECK-ERROR:         dup v5.2d, v1.b[1]
// CHECK-ERROR:                       ^

         dup v1.8b, b1
         dup v11.4h, h14
         dup v17.2s, s30
         dup v1.16b, d2
         dup v11.8s, w16
         dup v17.4d, w28
         dup v5.2d, w0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         dup v1.8b, b1
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         dup v11.4h, h14
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         dup v17.2s, s30
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         dup v1.16b, d2
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid vector kind qualifier
// CHECK-ERROR:         dup v11.8s, w16
// CHECK-ERROR:             ^
// CHECK-ERROR: error: invalid vector kind qualifier
// CHECK-ERROR:         dup v17.4d, w28
// CHECK-ERROR:             ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         dup v5.2d, w0
// CHECK-ERROR:                    ^

//----------------------------------------------------------------------
// Scalar Compare Bitwise Equal
//----------------------------------------------------------------------

         cmeq b20, d21, d22

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          cmeq b20, d21, d22
// CHECK-ERROR:               ^

//----------------------------------------------------------------------
// Scalar Compare Bitwise Equal To Zero
//----------------------------------------------------------------------

         cmeq d20, b21, #0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          cmeq d20, b21, #0
// CHECK-ERROR:                    ^

//----------------------------------------------------------------------
// Scalar Compare Unsigned Higher Or Same
//----------------------------------------------------------------------

         cmhs b20, d21, d22

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          cmhs b20, d21, d22
// CHECK-ERROR:               ^

        
//----------------------------------------------------------------------
// Scalar Compare Signed Greather Than Or Equal
//----------------------------------------------------------------------

         cmge b20, d21, d22

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          cmge b20, d21, d22
// CHECK-ERROR:               ^

//----------------------------------------------------------------------
// Scalar Compare Signed Greather Than Or Equal To Zero
//----------------------------------------------------------------------

         cmge d20, b21, #0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          cmge d20, b21, #0
// CHECK-ERROR:                    ^

//----------------------------------------------------------------------
// Scalar Compare Unsigned Higher
//----------------------------------------------------------------------

         cmhi b20, d21, d22

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          cmhi b20, d21, d22
// CHECK-ERROR:               ^

//----------------------------------------------------------------------
// Scalar Compare Signed Greater Than
//----------------------------------------------------------------------

         cmgt b20, d21, d22

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          cmgt b20, d21, d22
// CHECK-ERROR:               ^

//----------------------------------------------------------------------
// Scalar Compare Signed Greater Than Zero
//----------------------------------------------------------------------

         cmgt d20, b21, #0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          cmgt d20, b21, #0
// CHECK-ERROR:                    ^

//----------------------------------------------------------------------
// Scalar Compare Signed Less Than Or Equal To Zero
//----------------------------------------------------------------------

         cmle d20, b21, #0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          cmle d20, b21, #0
// CHECK-ERROR:                    ^

//----------------------------------------------------------------------
// Scalar Compare Less Than Zero
//----------------------------------------------------------------------

         cmlt d20, b21, #0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          cmlt d20, b21, #0
// CHECK-ERROR:                    ^

//----------------------------------------------------------------------
// Scalar Compare Bitwise Test Bits
//----------------------------------------------------------------------

         cmtst b20, d21, d22

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          cmtst b20, d21, d22
// CHECK-ERROR:                ^

//----------------------------------------------------------------------
// Scalar Floating-point Compare Mask Equal
//----------------------------------------------------------------------

         fcmeq s10, h11, s12
         fcmeq d20, s21, d22

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fcmeq s10, h11, s12
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fcmeq d20, s21, d22
// CHECK-ERROR:                     ^

//----------------------------------------------------------------------
// Scalar Floating-point Compare Mask Equal To Zero
//----------------------------------------------------------------------

         fcmeq h10, s11, #0.0
         fcmeq d20, s21, #0.0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fcmeq h10, s11, #0.0
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fcmeq d20, s21, #0.0
// CHECK-ERROR:                     ^

//----------------------------------------------------------------------
// Scalar Floating-point Compare Mask Greater Than Or Equal
//----------------------------------------------------------------------

         fcmge s10, h11, s12
         fcmge d20, s21, d22

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fcmge s10, h11, s12
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fcmge d20, s21, d22
// CHECK-ERROR:                     ^

//----------------------------------------------------------------------
// Scalar Floating-point Compare Mask Greater Than Or Equal To Zero
//----------------------------------------------------------------------

         fcmge h10, s11, #0.0
         fcmge d20, s21, #0.0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fcmge h10, s11, #0.0
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fcmge d20, s21, #0.0
// CHECK-ERROR:                     ^

//----------------------------------------------------------------------
// Scalar Floating-point Compare Mask Greather Than
//----------------------------------------------------------------------

         fcmgt s10, h11, s12
         fcmgt d20, s21, d22

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fcmgt s10, h11, s12
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fcmgt d20, s21, d22
// CHECK-ERROR:                     ^

//----------------------------------------------------------------------
// Scalar Floating-point Compare Mask Greather Than Zero
//----------------------------------------------------------------------

         fcmgt h10, s11, #0.0
         fcmgt d20, s21, #0.0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fcmgt h10, s11, #0.0
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fcmgt d20, s21, #0.0
// CHECK-ERROR:                     ^

//----------------------------------------------------------------------
// Scalar Floating-point Compare Mask Less Than Or Equal To Zero
//----------------------------------------------------------------------

         fcmle h10, s11, #0.0
         fcmle d20, s21, #0.0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fcmle h10, s11, #0.0
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fcmle d20, s21, #0.0
// CHECK-ERROR:                     ^

//----------------------------------------------------------------------
// Scalar Floating-point Compare Mask Less Than
//----------------------------------------------------------------------

         fcmlt h10, s11, #0.0
         fcmlt d20, s21, #0.0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fcmlt h10, s11, #0.0
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fcmlt d20, s21, #0.0
// CHECK-ERROR:                     ^

//----------------------------------------------------------------------
// Scalar Floating-point Absolute Compare Mask Greater Than Or Equal
//----------------------------------------------------------------------

         facge s10, h11, s12
         facge d20, s21, d22

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          facge s10, h11, s12
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          facge d20, s21, d22
// CHECK-ERROR:                     ^

//----------------------------------------------------------------------
// Scalar Floating-point Absolute Compare Mask Greater Than
//----------------------------------------------------------------------

         facgt s10, h11, s12
         facgt d20, d21, s22

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          facgt s10, h11, s12
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          facgt d20, d21, s22
// CHECK-ERROR:                          ^
        
//----------------------------------------------------------------------
// Scalar Signed Saturating Accumulated of Unsigned Value
//----------------------------------------------------------------------

        suqadd b0, h1
        suqadd h0, s1
        suqadd s0, d1
        suqadd d0, b0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        suqadd b0, h1
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        suqadd h0, s1
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        suqadd s0, d1
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        suqadd d0, b0
// CHECK-ERROR:                   ^

//----------------------------------------------------------------------
// Scalar Unsigned Saturating Accumulated of Signed Value
//----------------------------------------------------------------------

        usqadd b0, h1
        usqadd h0, s1
        usqadd s0, d1
        usqadd d0, b1

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        usqadd b0, h1
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        usqadd h0, s1
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        usqadd s0, d1
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        usqadd d0, b1
// CHECK-ERROR:                   ^

//----------------------------------------------------------------------
// Scalar Absolute Value
//----------------------------------------------------------------------

    abs d29, s24

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        abs d29, s24
// CHECK-ERROR:                 ^

//----------------------------------------------------------------------
// Scalar Negate
//----------------------------------------------------------------------

    neg d29, s24

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        neg d29, s24
// CHECK-ERROR:                 ^

//----------------------------------------------------------------------
// Signed Saturating Doubling Multiply-Add Long
//----------------------------------------------------------------------

    sqdmlal s17, h27, s12
    sqdmlal d19, s24, d12

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlal s17, h27, s12
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlal d19, s24, d12
// CHECK-ERROR:                          ^

//----------------------------------------------------------------------
// Signed Saturating Doubling Multiply-Subtract Long
//----------------------------------------------------------------------

    sqdmlsl s14, h12, s25
    sqdmlsl d12, s23, d13

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlsl s14, h12, s25
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmlsl d12, s23, d13
// CHECK-ERROR:                          ^

//----------------------------------------------------------------------
// Signed Saturating Doubling Multiply Long
//----------------------------------------------------------------------

    sqdmull s12, h22, s12
    sqdmull d15, s22, d12

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmull s12, h22, s12
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqdmull d15, s22, d12
// CHECK-ERROR:                          ^

//----------------------------------------------------------------------
// Scalar Signed Saturating Extract Unsigned Narrow
//----------------------------------------------------------------------

    sqxtun b19, b14
    sqxtun h21, h15
    sqxtun s20, s12

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqxtun b19, b14
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqxtun h21, h15
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqxtun s20, s12
// CHECK-ERROR:                    ^

//----------------------------------------------------------------------
// Scalar Signed Saturating Extract Signed Narrow
//----------------------------------------------------------------------

    sqxtn b18, b18
    sqxtn h20, h17
    sqxtn s19, s14

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqxtn b18, b18
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqxtn h20, h17
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sqxtn s19, s14
// CHECK-ERROR:                   ^


//----------------------------------------------------------------------
// Scalar Unsigned Saturating Extract Narrow
//----------------------------------------------------------------------

    uqxtn b18, b18
    uqxtn h20, h17
    uqxtn s19, s14

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uqxtn b18, b18
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uqxtn h20, h17
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        uqxtn s19, s14
// CHECK-ERROR:                   ^

//----------------------------------------------------------------------
// Scalar Signed Shift Right (Immediate)
//----------------------------------------------------------------------
        sshr d15, d16, #99

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:        sshr d15, d16, #99
// CHECK-ERROR:                       ^

        sshr d15, s16, #31

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        sshr d15, s16, #31
// CHECK-ERROR:                  ^

//----------------------------------------------------------------------
// Scalar Unsigned Shift Right (Immediate)
//----------------------------------------------------------------------

        ushr d10, d17, #99

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:        ushr d10, d17, #99
// CHECK-ERROR:                       ^

//----------------------------------------------------------------------
// Scalar Signed Rounding Shift Right (Immediate)
//----------------------------------------------------------------------

        srshr d19, d18, #99

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:        srshr d19, d18, #99
// CHECK-ERROR:                        ^

//----------------------------------------------------------------------
// Scalar Unigned Rounding Shift Right (Immediate)
//----------------------------------------------------------------------

        urshr d20, d23, #99

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:        urshr d20, d23, #99
// CHECK-ERROR:                        ^

//----------------------------------------------------------------------
// Scalar Signed Shift Right and Accumulate (Immediate)
//----------------------------------------------------------------------

        ssra d18, d12, #99

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:        ssra d18, d12, #99
// CHECK-ERROR:                       ^

//----------------------------------------------------------------------
// Scalar Unsigned Shift Right and Accumulate (Immediate)
//----------------------------------------------------------------------

        usra d20, d13, #99

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:        usra d20, d13, #99
// CHECK-ERROR:                       ^

//----------------------------------------------------------------------
// Scalar Signed Rounding Shift Right and Accumulate (Immediate)
//----------------------------------------------------------------------

        srsra d15, d11, #99

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:        srsra d15, d11, #99
// CHECK-ERROR:                        ^

//----------------------------------------------------------------------
// Scalar Unsigned Rounding Shift Right and Accumulate (Immediate)
//----------------------------------------------------------------------

        ursra d18, d10, #99

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:        ursra d18, d10, #99
// CHECK-ERROR:                        ^

//----------------------------------------------------------------------
// Scalar Shift Left (Immediate)
//----------------------------------------------------------------------

        shl d7, d10, #99

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR:        shl d7, d10, #99
// CHECK-ERROR:                     ^

        shl d7, s16, #31
        
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        shl d7, s16, #31
// CHECK-ERROR:                ^

//----------------------------------------------------------------------
// Signed Saturating Shift Left (Immediate)
//----------------------------------------------------------------------

        sqshl b11, b19, #99
        sqshl h13, h18, #99
        sqshl s14, s17, #99
        sqshl d15, d16, #99

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 7]
// CHECK-ERROR:        sqshl b11, b19, #99
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR:        sqshl h13, h18, #99
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR:        sqshl s14, s17, #99
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR:        sqshl d15, d16, #99
// CHECK-ERROR:                        ^

//----------------------------------------------------------------------
// Unsigned Saturating Shift Left (Immediate)
//----------------------------------------------------------------------

        uqshl b18, b15, #99
        uqshl h11, h18, #99
        uqshl s14, s19, #99
        uqshl d15, d12, #99

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 7]
// CHECK-ERROR:        uqshl b18, b15, #99
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR:        uqshl h11, h18, #99
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR:        uqshl s14, s19, #99
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR:        uqshl d15, d12, #99
// CHECK-ERROR:                        ^

//----------------------------------------------------------------------
// Signed Saturating Shift Left Unsigned (Immediate)
//----------------------------------------------------------------------

        sqshlu b15, b18, #99
        sqshlu h19, h17, #99
        sqshlu s16, s14, #99
        sqshlu d11, d13, #99

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 7]
// CHECK-ERROR:        sqshlu  b15, b18, #99
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 15]
// CHECK-ERROR:        sqshlu  h19, h17, #99
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 31]
// CHECK-ERROR:        sqshlu  s16, s14, #99
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR:        sqshlu  d11, d13, #99
// CHECK-ERROR:                          ^

//----------------------------------------------------------------------
// Shift Right And Insert (Immediate)
//----------------------------------------------------------------------

        sri d10, d12, #99

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:        sri d10, d12, #99
// CHECK-ERROR:                      ^

//----------------------------------------------------------------------
// Shift Left And Insert (Immediate)
//----------------------------------------------------------------------

        sli d10, d14, #99

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [0, 63]
// CHECK-ERROR:        sli d10, d14, #99
// CHECK-ERROR:                      ^

//----------------------------------------------------------------------
// Signed Saturating Shift Right Narrow (Immediate)
//----------------------------------------------------------------------

        sqshrn b10, h15, #99
        sqshrn h17, s10, #99
        sqshrn s18, d10, #99

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:        sqshrn  b10, h15, #99
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:        sqshrn  h17, s10, #99
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:        sqshrn  s18, d10, #99
// CHECK-ERROR:                          ^
        
//----------------------------------------------------------------------
// Unsigned Saturating Shift Right Narrow (Immediate)
//----------------------------------------------------------------------

        uqshrn b12, h10, #99
        uqshrn h10, s14, #99
        uqshrn s10, d12, #99

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:        uqshrn  b12, h10, #99
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:        uqshrn  h10, s14, #99
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:        uqshrn  s10, d12, #99
// CHECK-ERROR:                          ^
        
//----------------------------------------------------------------------
// Signed Saturating Rounded Shift Right Narrow (Immediate)
//----------------------------------------------------------------------

        sqrshrn b10, h13, #99
        sqrshrn h15, s10, #99
        sqrshrn s15, d12, #99

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:        sqrshrn b10, h13, #99
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:        sqrshrn h15, s10, #99
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:        sqrshrn s15, d12, #99
// CHECK-ERROR:                          ^
        
//----------------------------------------------------------------------
// Unsigned Saturating Rounded Shift Right Narrow (Immediate)
//----------------------------------------------------------------------

        uqrshrn b10, h12, #99
        uqrshrn h12, s10, #99
        uqrshrn s10, d10, #99

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:        uqrshrn b10, h12, #99
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:        uqrshrn h12, s10, #99
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:        uqrshrn s10, d10, #99
// CHECK-ERROR:                          ^

//----------------------------------------------------------------------
// Signed Saturating Shift Right Unsigned Narrow (Immediate)
//----------------------------------------------------------------------

        sqshrun b15, h10, #99
        sqshrun h20, s14, #99
        sqshrun s10, d15, #99

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:        sqshrun b15, h10, #99
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:        sqshrun h20, s14, #99
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:        sqshrun s10, d15, #99
// CHECK-ERROR:                          ^

//----------------------------------------------------------------------
// Signed Saturating Rounded Shift Right Unsigned Narrow (Immediate)
//----------------------------------------------------------------------

        sqrshrun b17, h10, #99
        sqrshrun h10, s13, #99
        sqrshrun s22, d16, #99

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 8]
// CHECK-ERROR:        sqrshrun b17, h10, #99
// CHECK-ERROR:                           ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 16]
// CHECK-ERROR:        sqrshrun h10, s13, #99
// CHECK-ERROR:                           ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:        sqrshrun s22, d16, #99
// CHECK-ERROR:                           ^

//----------------------------------------------------------------------
// Scalar Signed Fixed-point Convert To Floating-Point (Immediate)
//----------------------------------------------------------------------

    scvtf s22, s13, #0
    scvtf s22, s13, #33
    scvtf d21, d12, #65
    scvtf d21, s12, #31
        
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:        scvtf s22, s13, #0
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:        scvtf s22, s13, #33
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:        scvtf d21, d12, #65
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        scvtf d21, s12, #31
// CHECK-ERROR:                   ^

//----------------------------------------------------------------------
// Scalar Unsigned Fixed-point Convert To Floating-Point (Immediate)
//----------------------------------------------------------------------

    ucvtf s22, s13, #34
    ucvtf d21, d14, #65
    ucvtf d21, s14, #64
        
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:        ucvtf s22, s13, #34
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:        ucvtf d21, d14, #65
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        ucvtf d21, s14, #64
// CHECK-ERROR:                   ^

//------------------------------------------------------------------------------
// Element reverse
//------------------------------------------------------------------------------
         rev64 v6.2d, v8.2d
         rev32 v30.2s, v31.2s
         rev32 v30.4s, v31.4s
         rev32 v30.2d, v31.2d
         rev16 v21.4h, v1.4h
         rev16 v21.8h, v1.8h
         rev16 v21.2s, v1.2s
         rev16 v21.4s, v1.4s
         rev16 v21.2d, v1.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         rev64 v6.2d, v8.2d
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         rev32 v30.2s, v31.2s
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         rev32 v30.4s, v31.4s
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         rev32 v30.2d, v31.2d
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         rev16 v21.4h, v1.4h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         rev16 v21.8h, v1.8h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         rev16 v21.2s, v1.2s
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         rev16 v21.4s, v1.4s
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         rev16 v21.2d, v1.2d
// CHECK-ERROR:                   ^

//------------------------------------------------------------------------------
// Signed integer pairwise add long
//------------------------------------------------------------------------------

         saddlp v3.8h, v21.8h
         saddlp v8.8b, v5.8b
         saddlp v9.8h, v1.4s
         saddlp v0.4s, v1.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         saddlp v3.8h, v21.8h
// CHECK-ERROR:                           ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         saddlp v8.8b, v5.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         saddlp v9.8h, v1.4s
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         saddlp v0.4s, v1.2d
// CHECK-ERROR:                          ^

//------------------------------------------------------------------------------
// Unsigned integer pairwise add long
//------------------------------------------------------------------------------

         uaddlp v3.8h, v21.8h
         uaddlp v8.8b, v5.8b
         uaddlp v9.8h, v1.4s
         uaddlp v0.4s, v1.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uaddlp v3.8h, v21.8h
// CHECK-ERROR:                           ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uaddlp v8.8b, v5.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uaddlp v9.8h, v1.4s
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uaddlp v0.4s, v1.2d
// CHECK-ERROR:                          ^

//------------------------------------------------------------------------------
// Signed integer pairwise add and accumulate long
//------------------------------------------------------------------------------

         sadalp v3.16b, v21.16b
         sadalp v8.4h, v5.4h
         sadalp v9.4s, v1.4s
         sadalp v0.4h, v1.2s
         sadalp v12.2d, v4.8h

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sadalp v3.16b, v21.16b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sadalp v8.4h, v5.4h
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sadalp v9.4s, v1.4s
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sadalp v0.4h, v1.2s
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sadalp v12.2d, v4.8h
// CHECK-ERROR:                           ^

//------------------------------------------------------------------------------
// Unsigned integer pairwise add and accumulate long
//------------------------------------------------------------------------------

         uadalp v3.16b, v21.16b
         uadalp v8.4h, v5.4h
         uadalp v9.4s, v1.4s
         uadalp v0.4h, v1.2s
         uadalp v12.2d, v4.8h

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uadalp v3.16b, v21.16b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uadalp v8.4h, v5.4h
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uadalp v9.4s, v1.4s
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uadalp v0.4h, v1.2s
// CHECK-ERROR:                          ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uadalp v12.2d, v4.8h
// CHECK-ERROR:                           ^

//------------------------------------------------------------------------------
// Signed integer saturating accumulate of unsigned value
//------------------------------------------------------------------------------

         suqadd v0.16b, v31.8b
         suqadd v1.8b, v9.8h
         suqadd v13.4h, v21.4s
         suqadd v4.2s, v0.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         suqadd v0.16b, v31.8b
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         suqadd v1.8b, v9.8h
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         suqadd v13.4h, v21.4s
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         suqadd v4.2s, v0.2d
// CHECK-ERROR:                       ^

//------------------------------------------------------------------------------
// Unsigned integer saturating accumulate of signed value
//------------------------------------------------------------------------------

         usqadd v0.16b, v31.8b
         usqadd v2.8h, v4.4h
         usqadd v13.4h, v21.4s
         usqadd v4.2s, v0.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         usqadd v0.16b, v31.8b
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         usqadd v2.8h, v4.4h
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         usqadd v13.4h, v21.4s
// CHECK-ERROR:                        ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         usqadd v4.2s, v0.2d
// CHECK-ERROR:                       ^

//------------------------------------------------------------------------------
// Integer saturating absolute
//------------------------------------------------------------------------------

         sqabs v0.16b, v31.8b
         sqabs v2.8h, v4.4h
         sqabs v6.4s, v8.2s
         sqabs v6.2d, v8.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqabs v0.16b, v31.8b
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqabs v2.8h, v4.4h
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqabs v6.4s, v8.2s
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqabs v6.2d, v8.2s
// CHECK-ERROR:                      ^

//------------------------------------------------------------------------------
// Signed integer saturating negate
//------------------------------------------------------------------------------

         sqneg v0.16b, v31.8b
         sqneg v2.8h, v4.4h
         sqneg v6.4s, v8.2s
         sqneg v6.2d, v8.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqneg v0.16b, v31.8b
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqneg v2.8h, v4.4h
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqneg v6.4s, v8.2s
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqneg v6.2d, v8.2s
// CHECK-ERROR:                      ^

//------------------------------------------------------------------------------
// Integer absolute
//------------------------------------------------------------------------------

         abs v0.16b, v31.8b
         abs v2.8h, v4.4h
         abs v6.4s, v8.2s
         abs v6.2d, v8.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         abs v0.16b, v31.8b
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         abs v2.8h, v4.4h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         abs v6.4s, v8.2s
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         abs v6.2d, v8.2s
// CHECK-ERROR:                    ^

//------------------------------------------------------------------------------
// Integer count leading sign bits
//------------------------------------------------------------------------------

         cls v0.2d, v31.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         cls v0.2d, v31.2d
// CHECK-ERROR:                ^

//------------------------------------------------------------------------------
// Integer count leading zeros
//------------------------------------------------------------------------------

         clz v0.2d, v31.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         clz v0.2d, v31.2d
// CHECK-ERROR:                ^

//------------------------------------------------------------------------------
// Population count
//------------------------------------------------------------------------------

         cnt v2.8h, v4.8h
         cnt v6.4s, v8.4s
         cnt v6.2d, v8.2d
         cnt v13.4h, v21.4h
         cnt v4.2s, v0.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         cnt v2.8h, v4.8h
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         cnt v6.4s, v8.4s
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         cnt v6.2d, v8.2d
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         cnt v13.4h, v21.4h
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         cnt v4.2s, v0.2s
// CHECK-ERROR:                ^


//------------------------------------------------------------------------------
// Bitwise NOT
//------------------------------------------------------------------------------

         not v2.8h, v4.8h
         not v6.4s, v8.4s
         not v6.2d, v8.2d
         not v13.4h, v21.4h
         not v4.2s, v0.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         not v2.8h, v4.8h
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         not v6.4s, v8.4s
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         not v6.2d, v8.2d
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         not v13.4h, v21.4h
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         not v4.2s, v0.2s
// CHECK-ERROR:                ^

//------------------------------------------------------------------------------
// Bitwise reverse
//------------------------------------------------------------------------------

         rbit v2.8h, v4.8h
         rbit v6.4s, v8.4s
         rbit v6.2d, v8.2d
         rbit v13.4h, v21.4h
         rbit v4.2s, v0.2s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         rbit v2.8h, v4.8h
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         rbit v6.4s, v8.4s
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         rbit v6.2d, v8.2d
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         rbit v13.4h, v21.4h
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         rbit v4.2s, v0.2s
// CHECK-ERROR:                 ^

//------------------------------------------------------------------------------
// Floating-point absolute
//------------------------------------------------------------------------------

         fabs v0.16b, v31.16b
         fabs v2.8h, v4.8h
         fabs v1.8b, v9.8b
         fabs v13.4h, v21.4h

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fabs v0.16b, v31.16b
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fabs v2.8h, v4.8h
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fabs v1.8b, v9.8b
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fabs v13.4h, v21.4h
// CHECK-ERROR:                  ^

//------------------------------------------------------------------------------
// Floating-point negate
//------------------------------------------------------------------------------

         fneg v0.16b, v31.16b
         fneg v2.8h, v4.8h
         fneg v1.8b, v9.8b
         fneg v13.4h, v21.4h

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fneg v0.16b, v31.16b
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fneg v2.8h, v4.8h
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fneg v1.8b, v9.8b
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fneg v13.4h, v21.4h
// CHECK-ERROR:                  ^

//------------------------------------------------------------------------------
// Integer extract and narrow
//------------------------------------------------------------------------------

         xtn v0.16b, v31.8h
         xtn v2.8h, v4.4s
         xtn v6.4s, v8.2d
         xtn2 v1.8b, v9.8h
         xtn2 v13.4h, v21.4s
         xtn2 v4.2s, v0.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         xtn v0.16b, v31.8h
// CHECK-ERROR:             ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         xtn v2.8h, v4.4s
// CHECK-ERROR:             ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         xtn v6.4s, v8.2d
// CHECK-ERROR:             ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         xtn2 v1.8b, v9.8h
// CHECK-ERROR:              ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         xtn2 v13.4h, v21.4s
// CHECK-ERROR:              ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         xtn2 v4.2s, v0.2d
// CHECK-ERROR:              ^

//------------------------------------------------------------------------------
// Signed integer saturating extract and unsigned narrow
//------------------------------------------------------------------------------

         sqxtun v0.16b, v31.8h
         sqxtun v2.8h, v4.4s
         sqxtun v6.4s, v8.2d
         sqxtun2 v1.8b, v9.8h
         sqxtun2 v13.4h, v21.4s
         sqxtun2 v4.2s, v0.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqxtun v0.16b, v31.8h
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqxtun v2.8h, v4.4s
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqxtun v6.4s, v8.2d
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqxtun2 v1.8b, v9.8h
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqxtun2 v13.4h, v21.4s
// CHECK-ERROR:                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqxtun2 v4.2s, v0.2d
// CHECK-ERROR:                 ^

//------------------------------------------------------------------------------
// Signed integer saturating extract and narrow
//------------------------------------------------------------------------------

         sqxtn v0.16b, v31.8h
         sqxtn v2.8h, v4.4s
         sqxtn v6.4s, v8.2d
         sqxtn2 v1.8b, v9.8h
         sqxtn2 v13.4h, v21.4s
         sqxtn2 v4.2s, v0.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqxtn v0.16b, v31.8h
// CHECK-ERROR:               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqxtn v2.8h, v4.4s
// CHECK-ERROR:               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqxtn v6.4s, v8.2d
// CHECK-ERROR:               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqxtn2 v1.8b, v9.8h
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqxtn2 v13.4h, v21.4s
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         sqxtn2 v4.2s, v0.2d
// CHECK-ERROR:                ^

//------------------------------------------------------------------------------
// Unsigned integer saturating extract and narrow
//------------------------------------------------------------------------------

         uqxtn v0.16b, v31.8h
         uqxtn v2.8h, v4.4s
         uqxtn v6.4s, v8.2d
         uqxtn2 v1.8b, v9.8h
         uqxtn2 v13.4h, v21.4s
         uqxtn2 v4.2s, v0.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uqxtn v0.16b, v31.8h
// CHECK-ERROR:               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uqxtn v2.8h, v4.4s
// CHECK-ERROR:               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uqxtn v6.4s, v8.2d
// CHECK-ERROR:               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uqxtn2 v1.8b, v9.8h
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uqxtn2 v13.4h, v21.4s
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         uqxtn2 v4.2s, v0.2d
// CHECK-ERROR:                ^

//------------------------------------------------------------------------------
// Integer shift left long
//------------------------------------------------------------------------------

         shll2 v2.8h, v4.16b, #7
         shll2 v6.4s, v8.8h, #15
         shll2 v6.2d, v8.4s, #31
         shll v2.8h, v4.16b, #8
         shll v6.4s, v8.8h, #16
         shll v6.2d, v8.4s, #32
         shll v2.8h, v4.8b, #8
         shll v6.4s, v8.4h, #16
         shll v6.2d, v8.2s, #32
         shll2 v2.8h, v4.8b, #5
         shll2 v6.4s, v8.4h, #14
         shll2 v6.2d, v8.2s, #1

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         shll2 v2.8h, v4.16b, #7
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         shll2 v6.4s, v8.8h, #15
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         shll2 v6.2d, v8.4s, #31
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         shll v2.8h, v4.16b, #8
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         shll v6.4s, v8.8h, #16
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         shll v6.2d, v8.4s, #32
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         shll2 v2.8h, v4.8b, #5
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         shll2 v6.4s, v8.4h, #14
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         shll2 v6.2d, v8.2s, #1
// CHECK-ERROR:                      ^

//------------------------------------------------------------------------------
// Floating-point convert downsize
//------------------------------------------------------------------------------

         fcvtn v2.8h, v4.4s
         fcvtn v6.4s, v8.2d
         fcvtn2 v13.4h, v21.4s
         fcvtn2 v4.2s, v0.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtn v2.8h, v4.4s
// CHECK-ERROR:               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtn v6.4s, v8.2d
// CHECK-ERROR:               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtn2 v13.4h, v21.4s
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtn2 v4.2s, v0.2d
// CHECK-ERROR:                ^

//------------------------------------------------------------------------------
// Floating-point convert downsize with inexact
//------------------------------------------------------------------------------

         fcvtxn v6.4s, v8.2d
         fcvtxn2 v4.2s, v0.2d

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtxn v6.4s, v8.2d
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtxn2 v4.2s, v0.2d
// CHECK-ERROR:                 ^

//------------------------------------------------------------------------------
// Floating-point convert upsize
//------------------------------------------------------------------------------

         fcvtl2 v9.4s, v1.4h
         fcvtl2 v0.2d, v1.2s
         fcvtl v12.4s, v4.8h
         fcvtl v17.2d, v28.4s

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtl2 v9.4s, v1.4h
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtl2 v0.2d, v1.2s
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtl v12.4s, v4.8h
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtl v17.2d, v28.4s
// CHECK-ERROR:                       ^

//------------------------------------------------------------------------------
// Floating-point round to integral
//------------------------------------------------------------------------------

         frintn v0.16b, v31.16b
         frintn v2.8h, v4.8h
         frintn v1.8b, v9.8b
         frintn v13.4h, v21.4h

         frinta v0.16b, v31.16b
         frinta v2.8h, v4.8h
         frinta v1.8b, v9.8b
         frinta v13.4h, v21.4h

         frintp v0.16b, v31.16b
         frintp v2.8h, v4.8h
         frintp v1.8b, v9.8b
         frintp v13.4h, v21.4h

         frintm v0.16b, v31.16b
         frintm v2.8h, v4.8h
         frintm v1.8b, v9.8b
         frintm v13.4h, v21.4h

         frintx v0.16b, v31.16b
         frintx v2.8h, v4.8h
         frintx v1.8b, v9.8b
         frintx v13.4h, v21.4h

         frintz v0.16b, v31.16b
         frintz v2.8h, v4.8h
         frintz v1.8b, v9.8b
         frintz v13.4h, v21.4h

         frinti v0.16b, v31.16b
         frinti v2.8h, v4.8h
         frinti v1.8b, v9.8b
         frinti v13.4h, v21.4h

         fcvtns v0.16b, v31.16b
         fcvtns v2.8h, v4.8h
         fcvtns v1.8b, v9.8b
         fcvtns v13.4h, v21.4h

         fcvtnu v0.16b, v31.16b
         fcvtnu v2.8h, v4.8h
         fcvtnu v1.8b, v9.8b
         fcvtnu v13.4h, v21.4h

         fcvtps v0.16b, v31.16b
         fcvtps v2.8h, v4.8h
         fcvtps v1.8b, v9.8b
         fcvtps v13.4h, v21.4h

         fcvtpu v0.16b, v31.16b
         fcvtpu v2.8h, v4.8h
         fcvtpu v1.8b, v9.8b
         fcvtpu v13.4h, v21.4h

         fcvtms v0.16b, v31.16b
         fcvtms v2.8h, v4.8h
         fcvtms v1.8b, v9.8b
         fcvtms v13.4h, v21.4h

         fcvtmu v0.16b, v31.16b
         fcvtmu v2.8h, v4.8h
         fcvtmu v1.8b, v9.8b
         fcvtmu v13.4h, v21.4h

         fcvtzs v0.16b, v31.16b
         fcvtzs v2.8h, v4.8h
         fcvtzs v1.8b, v9.8b
         fcvtzs v13.4h, v21.4h

         fcvtzu v0.16b, v31.16b
         fcvtzu v2.8h, v4.8h
         fcvtzu v1.8b, v9.8b
         fcvtzu v13.4h, v21.4h

         fcvtas v0.16b, v31.16b
         fcvtas v2.8h, v4.8h
         fcvtas v1.8b, v9.8b
         fcvtas v13.4h, v21.4h

         fcvtau v0.16b, v31.16b
         fcvtau v2.8h, v4.8h
         fcvtau v1.8b, v9.8b
         fcvtau v13.4h, v21.4h

         urecpe v0.16b, v31.16b
         urecpe v2.8h, v4.8h
         urecpe v1.8b, v9.8b
         urecpe v13.4h, v21.4h
         urecpe v1.2d, v9.2d

         ursqrte v0.16b, v31.16b
         ursqrte v2.8h, v4.8h
         ursqrte v1.8b, v9.8b
         ursqrte v13.4h, v21.4h
         ursqrte v1.2d, v9.2d

         scvtf v0.16b, v31.16b
         scvtf v2.8h, v4.8h
         scvtf v1.8b, v9.8b
         scvtf v13.4h, v21.4h

         ucvtf v0.16b, v31.16b
         ucvtf v2.8h, v4.8h
         ucvtf v1.8b, v9.8b
         ucvtf v13.4h, v21.4h

         frecpe v0.16b, v31.16b
         frecpe v2.8h, v4.8h
         frecpe v1.8b, v9.8b
         frecpe v13.4h, v21.4h

         frsqrte v0.16b, v31.16b
         frsqrte v2.8h, v4.8h
         frsqrte v1.8b, v9.8b
         frsqrte v13.4h, v21.4h

         fsqrt v0.16b, v31.16b
         fsqrt v2.8h, v4.8h
         fsqrt v1.8b, v9.8b
         fsqrt v13.4h, v21.4h

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         frintn v0.16b, v31.16b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         frintn v2.8h, v4.8h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         frintn v1.8b, v9.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         frintn v13.4h, v21.4h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         frinta v0.16b, v31.16b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         frinta v2.8h, v4.8h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         frinta v1.8b, v9.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         frinta v13.4h, v21.4h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         frintp v0.16b, v31.16b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         frintp v2.8h, v4.8h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         frintp v1.8b, v9.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         frintp v13.4h, v21.4h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         frintm v0.16b, v31.16b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         frintm v2.8h, v4.8h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         frintm v1.8b, v9.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         frintm v13.4h, v21.4h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         frintx v0.16b, v31.16b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         frintx v2.8h, v4.8h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         frintx v1.8b, v9.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         frintx v13.4h, v21.4h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         frintz v0.16b, v31.16b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         frintz v2.8h, v4.8h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         frintz v1.8b, v9.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         frintz v13.4h, v21.4h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         frinti v0.16b, v31.16b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         frinti v2.8h, v4.8h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         frinti v1.8b, v9.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         frinti v13.4h, v21.4h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtns v0.16b, v31.16b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fcvtns v2.8h, v4.8h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtns v1.8b, v9.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fcvtns v13.4h, v21.4h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtnu v0.16b, v31.16b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fcvtnu v2.8h, v4.8h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtnu v1.8b, v9.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fcvtnu v13.4h, v21.4h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtps v0.16b, v31.16b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fcvtps v2.8h, v4.8h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtps v1.8b, v9.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fcvtps v13.4h, v21.4h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtpu v0.16b, v31.16b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fcvtpu v2.8h, v4.8h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtpu v1.8b, v9.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fcvtpu v13.4h, v21.4h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtms v0.16b, v31.16b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fcvtms v2.8h, v4.8h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtms v1.8b, v9.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fcvtms v13.4h, v21.4h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtmu v0.16b, v31.16b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fcvtmu v2.8h, v4.8h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtmu v1.8b, v9.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fcvtmu v13.4h, v21.4h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtzs v0.16b, v31.16b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fcvtzs v2.8h, v4.8h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtzs v1.8b, v9.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fcvtzs v13.4h, v21.4h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtzu v0.16b, v31.16b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fcvtzu v2.8h, v4.8h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtzu v1.8b, v9.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fcvtzu v13.4h, v21.4h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtas v0.16b, v31.16b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fcvtas v2.8h, v4.8h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtas v1.8b, v9.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fcvtas v13.4h, v21.4h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtau v0.16b, v31.16b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fcvtau v2.8h, v4.8h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fcvtau v1.8b, v9.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fcvtau v13.4h, v21.4h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         urecpe v0.16b, v31.16b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         urecpe v2.8h, v4.8h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         urecpe v1.8b, v9.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         urecpe v13.4h, v21.4h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         urecpe v1.2d, v9.2d
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ursqrte v0.16b, v31.16b
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ursqrte v2.8h, v4.8h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ursqrte v1.8b, v9.8b
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ursqrte v13.4h, v21.4h
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ursqrte v1.2d, v9.2d
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         scvtf v0.16b, v31.16b
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         scvtf v2.8h, v4.8h
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         scvtf v1.8b, v9.8b
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         scvtf v13.4h, v21.4h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ucvtf v0.16b, v31.16b
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         ucvtf v2.8h, v4.8h
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ucvtf v1.8b, v9.8b
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         ucvtf v13.4h, v21.4h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         frecpe v0.16b, v31.16b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         frecpe v2.8h, v4.8h
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         frecpe v1.8b, v9.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         frecpe v13.4h, v21.4h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         frsqrte v0.16b, v31.16b
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         frsqrte v2.8h, v4.8h
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         frsqrte v1.8b, v9.8b
// CHECK-ERROR:                    ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         frsqrte v13.4h, v21.4h
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fsqrt v0.16b, v31.16b
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fsqrt v2.8h, v4.8h
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         fsqrt v1.8b, v9.8b
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: instruction requires: fullfp16
// CHECK-ERROR:         fsqrt v13.4h, v21.4h
// CHECK-ERROR:                   ^

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Signed Fixed-point (Immediate)
//----------------------------------------------------------------------

    fcvtzs s21, s12, #0
    fcvtzs d21, d12, #65
    fcvtzs s21, d12, #1

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:        fcvtzs s21, s12, #0
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:        fcvtzs d21, d12, #65
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtzs s21, d12, #1
// CHECK-ERROR:                    ^

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Unsigned Fixed-point (Immediate)
//----------------------------------------------------------------------

    fcvtzu s21, s12, #33
    fcvtzu d21, d12, #0
    fcvtzu s21, d12, #1

// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 32]
// CHECK-ERROR:        fcvtzu s21, s12, #33
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: {{expected|immediate must be an}} integer in range [1, 64]
// CHECK-ERROR:        fcvtzu d21, d12, #0
// CHECK-ERROR:                         ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtzu s21, d12, #1
// CHECK-ERROR:                    ^

//----------------------------------------------------------------------
// Scalar Unsigned Saturating Extract Narrow
//----------------------------------------------------------------------

        aese v0.8h, v1.8h
        aese v0.4s, v1.4s
        aese v0.2d, v1.2d
        aesd v0.8h, v1.8h
        aesmc v0.8h, v1.8h
        aesimc v0.8h, v1.8h

// CHECK:  error: invalid operand for instruction
// CHECK:         aese v0.8h, v1.8h
// CHECK:                 ^
// CHECK:  error: invalid operand for instruction
// CHECK:         aese v0.4s, v1.4s
// CHECK:                 ^
// CHECK:  error: invalid operand for instruction
// CHECK:         aese v0.2d, v1.2d
// CHECK:                 ^
// CHECK:  error: invalid operand for instruction
// CHECK:         aesd v0.8h, v1.8h
// CHECK:                 ^
// CHECK:  error: invalid operand for instruction
// CHECK:         aesmc v0.8h, v1.8h
// CHECK:                  ^
// CHECK:  error: invalid operand for instruction
// CHECK:         aesimc v0.8h, v1.8h
// CHECK:                   ^

        sha1h b0, b1
        sha1h h0, h1
        sha1h d0, d1
        sha1h q0, q1
        sha1su1 v0.16b, v1.16b
        sha1su1 v0.8h, v1.8h
        sha1su1 v0.2d, v1.2d
        sha256su0 v0.16b, v1.16b

// CHECK:  error: invalid operand for instruction
// CHECK:         sha1h b0, b1
// CHECK:               ^
// CHECK:  error: invalid operand for instruction
// CHECK:         sha1h h0, h1
// CHECK:               ^
// CHECK:  error: invalid operand for instruction
// CHECK:         sha1h d0, d1
// CHECK:               ^
// CHECK:  error: invalid operand for instruction
// CHECK:         sha1h q0, q1
// CHECK:               ^
// CHECK:  error: invalid operand for instruction
// CHECK:         sha1su1 v0.16b, v1.16b
// CHECK:                    ^
// CHECK:  error: invalid operand for instruction
// CHECK:         sha1su1 v0.8h, v1.8h
// CHECK:                    ^
// CHECK:  error: invalid operand for instruction
// CHECK:         sha1su1 v0.2d, v1.2d
// CHECK:                    ^
// CHECK:  error: invalid operand for instruction
// CHECK:         sha256su0 v0.16b, v1.16b
// CHECK:                      ^

        sha1c q0, q1, v2.4s
        sha1p q0, q1, v2.4s
        sha1m q0, q1, v2.4s
        sha1su0 v0.16b, v1.16b, v2.16b
        sha1su0 v0.8h, v1.8h, v2.8h
        sha1su0 v0.2d, v1.2d, v2.2d
        sha256h q0, q1, q2
        sha256h v0.4s, v1.4s, v2.4s
        sha256h2 q0, q1, q2
        sha256su1 v0.16b, v1.16b, v2.16b

// CHECK:  error: invalid operand for instruction
// CHECK:         sha1c q0, q1, v2.4s
// CHECK:                   ^
// CHECK:  error: invalid operand for instruction
// CHECK:         sha1p q0, q1, v2.4s
// CHECK:                   ^
// CHECK:  error: invalid operand for instruction
// CHECK:         sha1m q0, q1, v2.4s
// CHECK:                   ^
// CHECK:  error: invalid operand for instruction
// CHECK:         sha1su0 v0.16b, v1.16b, v2.16b
// CHECK:                    ^
// CHECK:  error: invalid operand for instruction
// CHECK:         sha1su0 v0.8h, v1.8h, v2.8h
// CHECK:                    ^
// CHECK:  error: invalid operand for instruction
// CHECK:         sha1su0 v0.2d, v1.2d, v2.2d
// CHECK:                    ^
// CHECK:  error: too few operands for instruction
// CHECK:         sha256h q0, q1, q2
// CHECK:         ^
// CHECK:  error: invalid operand for instruction
// CHECK:         sha256h v0.4s, v1.4s, v2.4s
// CHECK:                    ^
// CHECK:  error: too few operands for instruction
// CHECK:         sha256h2 q0, q1, q2
// CHECK:         ^
// CHECK:  error: invalid operand for instruction
// CHECK:         sha256su1 v0.16b, v1.16b, v2.16b
// CHECK:                      ^

//----------------------------------------------------------------------
// Bitwise extract
//----------------------------------------------------------------------

        ext v0.8b, v1.8b, v2.4h, #0x3
        ext v0.4h, v1.4h, v2.4h, #0x3
        ext v0.2s, v1.2s, v2.2s, #0x1
        ext v0.1d, v1.1d, v2.1d, #0x0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ext v0.8b, v1.8b, v2.4h, #0x3
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ext v0.4h, v1.4h, v2.4h, #0x3
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ext v0.2s, v1.2s, v2.2s, #0x1
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ext v0.1d, v1.1d, v2.1d, #0x0
// CHECK-ERROR:                ^

        ext v0.16b, v1.16b, v2.8h, #0x3
        ext v0.8h, v1.8h, v2.8h, #0x3
        ext v0.4s, v1.4s, v2.4s, #0x1
        ext v0.2d, v1.2d, v2.2d, #0x0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ext v0.16b, v1.16b, v2.8h, #0x3
// CHECK-ERROR:                                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ext v0.8h, v1.8h, v2.8h, #0x3
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ext v0.4s, v1.4s, v2.4s, #0x1
// CHECK-ERROR:                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:         ext v0.2d, v1.2d, v2.2d, #0x0
// CHECK-ERROR:                ^


//----------------------------------------------------------------------
// Permutation with 3 vectors
//----------------------------------------------------------------------

        uzp1 v0.16b, v1.8b, v2.8b
// CHECK-ERROR: [[@LINE-1]]:22: error: invalid operand for instruction
        uzp1 v0.8b, v1.4b, v2.4b
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        uzp1 v0.8h, v1.4h, v2.4h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        uzp1 v0.4h, v1.2h, v2.2h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        uzp1 v0.4s, v1.2s, v2.2s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        uzp1 v0.2s, v1.1s, v2.1s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid vector kind qualifier
// CHECK-ERROR: [[@LINE-2]]:28: error: invalid vector kind qualifier
        uzp1 v0.2d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        uzp1 v0.1d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:14: error: invalid operand for instruction



        uzp2 v0.16b, v1.8b, v2.8b
// CHECK-ERROR: [[@LINE-1]]:22: error: invalid operand for instruction
        uzp2 v0.8b, v1.4b, v2.4b
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        uzp2 v0.8h, v1.4h, v2.4h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        uzp2 v0.4h, v1.2h, v2.2h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        uzp2 v0.4s, v1.2s, v2.2s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        uzp2 v0.2s, v1.1s, v2.1s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid vector kind qualifier
// CHECK-ERROR: [[@LINE-2]]:28: error: invalid vector kind qualifier
        uzp2 v0.2d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        uzp2 v0.1d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:14: error: invalid operand for instruction



        zip1 v0.16b, v1.8b, v2.8b
// CHECK-ERROR: [[@LINE-1]]:22: error: invalid operand for instruction
        zip1 v0.8b, v1.4b, v2.4b
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        zip1 v0.8h, v1.4h, v2.4h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        zip1 v0.4h, v1.2h, v2.2h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        zip1 v0.4s, v1.2s, v2.2s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        zip1 v0.2s, v1.1s, v2.1s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid vector kind qualifier
// CHECK-ERROR: [[@LINE-2]]:28: error: invalid vector kind qualifier
        zip1 v0.2d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        zip1 v0.1d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:14: error: invalid operand for instruction



        zip2 v0.16b, v1.8b, v2.8b
// CHECK-ERROR: [[@LINE-1]]:22: error: invalid operand for instruction
        zip2 v0.8b, v1.4b, v2.4b
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        zip2 v0.8h, v1.4h, v2.4h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        zip2 v0.4h, v1.2h, v2.2h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        zip2 v0.4s, v1.2s, v2.2s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        zip2 v0.2s, v1.1s, v2.1s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid vector kind qualifier
// CHECK-ERROR: [[@LINE-2]]:28: error: invalid vector kind qualifier
        zip2 v0.2d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        zip2 v0.1d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:14: error: invalid operand for instruction



        trn1 v0.16b, v1.8b, v2.8b
// CHECK-ERROR: [[@LINE-1]]:22: error: invalid operand for instruction
        trn1 v0.8b, v1.4b, v2.4b
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        trn1 v0.8h, v1.4h, v2.4h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        trn1 v0.4h, v1.2h, v2.2h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        trn1 v0.4s, v1.2s, v2.2s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        trn1 v0.2s, v1.1s, v2.1s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid vector kind qualifier
// CHECK-ERROR: [[@LINE-2]]:28: error: invalid vector kind qualifier
        trn1 v0.2d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        trn1 v0.1d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:14: error: invalid operand for instruction



        trn2 v0.16b, v1.8b, v2.8b
// CHECK-ERROR: [[@LINE-1]]:22: error: invalid operand for instruction
        trn2 v0.8b, v1.4b, v2.4b
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        trn2 v0.8h, v1.4h, v2.4h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        trn2 v0.4h, v1.2h, v2.2h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        trn2 v0.4s, v1.2s, v2.2s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        trn2 v0.2s, v1.1s, v2.1s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid vector kind qualifier
// CHECK-ERROR: [[@LINE-2]]:28: error: invalid vector kind qualifier
        trn2 v0.2d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        trn2 v0.1d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:14: error: invalid operand for instruction

//----------------------------------------------------------------------
// Permutation with 3 vectors
//----------------------------------------------------------------------

        uzp1 v0.16b, v1.8b, v2.8b
// CHECK-ERROR: [[@LINE-1]]:22: error: invalid operand for instruction
        uzp1 v0.8b, v1.4b, v2.4b
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        uzp1 v0.8h, v1.4h, v2.4h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        uzp1 v0.4h, v1.2h, v2.2h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        uzp1 v0.4s, v1.2s, v2.2s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        uzp1 v0.2s, v1.1s, v2.1s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid vector kind qualifier
// CHECK-ERROR: [[@LINE-2]]:28: error: invalid vector kind qualifier
        uzp1 v0.2d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        uzp1 v0.1d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:14: error: invalid operand for instruction

        uzp2 v0.16b, v1.8b, v2.8b
// CHECK-ERROR: [[@LINE-1]]:22: error: invalid operand for instruction
        uzp2 v0.8b, v1.4b, v2.4b
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        uzp2 v0.8h, v1.4h, v2.4h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        uzp2 v0.4h, v1.2h, v2.2h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        uzp2 v0.4s, v1.2s, v2.2s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        uzp2 v0.2s, v1.1s, v2.1s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid vector kind qualifier
// CHECK-ERROR: [[@LINE-2]]:28: error: invalid vector kind qualifier
        uzp2 v0.2d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        uzp2 v0.1d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:14: error: invalid operand for instruction

        zip1 v0.16b, v1.8b, v2.8b
// CHECK-ERROR: [[@LINE-1]]:22: error: invalid operand for instruction
        zip1 v0.8b, v1.4b, v2.4b
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        zip1 v0.8h, v1.4h, v2.4h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        zip1 v0.4h, v1.2h, v2.2h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        zip1 v0.4s, v1.2s, v2.2s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        zip1 v0.2s, v1.1s, v2.1s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid vector kind qualifier
// CHECK-ERROR: [[@LINE-2]]:28: error: invalid vector kind qualifier
        zip1 v0.2d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        zip1 v0.1d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:14: error: invalid operand for instruction





        zip2 v0.16b, v1.8b, v2.8b
// CHECK-ERROR: [[@LINE-1]]:22: error: invalid operand for instruction
        zip2 v0.8b, v1.4b, v2.4b
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        zip2 v0.8h, v1.4h, v2.4h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        zip2 v0.4h, v1.2h, v2.2h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        zip2 v0.4s, v1.2s, v2.2s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        zip2 v0.2s, v1.1s, v2.1s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid vector kind qualifier
// CHECK-ERROR: [[@LINE-2]]:28: error: invalid vector kind qualifier
        zip2 v0.2d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        zip2 v0.1d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:14: error: invalid operand for instruction




        trn1 v0.16b, v1.8b, v2.8b
// CHECK-ERROR: [[@LINE-1]]:22: error: invalid operand for instruction
        trn1 v0.8b, v1.4b, v2.4b
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        trn1 v0.8h, v1.4h, v2.4h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        trn1 v0.4h, v1.2h, v2.2h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        trn1 v0.4s, v1.2s, v2.2s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        trn1 v0.2s, v1.1s, v2.1s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid vector kind qualifier
// CHECK-ERROR: [[@LINE-2]]:28: error: invalid vector kind qualifier
        trn1 v0.2d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        trn1 v0.1d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:14: error: invalid operand for instruction



        trn2 v0.16b, v1.8b, v2.8b
// CHECK-ERROR: [[@LINE-1]]:22: error: invalid operand for instruction
        trn2 v0.8b, v1.4b, v2.4b
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        trn2 v0.8h, v1.4h, v2.4h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        trn2 v0.4h, v1.2h, v2.2h
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        trn2 v0.4s, v1.2s, v2.2s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        trn2 v0.2s, v1.1s, v2.1s
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid vector kind qualifier
// CHECK-ERROR: [[@LINE-2]]:28: error: invalid vector kind qualifier
        trn2 v0.2d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:21: error: invalid operand for instruction
        trn2 v0.1d, v1.1d, v2.1d
// CHECK-ERROR: [[@LINE-1]]:14: error: invalid operand for instruction



//----------------------------------------------------------------------
// Floating Point  multiply (scalar, by element)
//----------------------------------------------------------------------
      // mismatched and invalid vector types
      fmul    s0, s1, v1.h[0]
      fmul    h0, h1, v1.s[0]
      // invalid lane
      fmul    s2, s29, v10.s[4]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fmul    s0, s1, v1.h[0]
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fmul    h0, h1, v1.s[0]
// CHECK-ERROR:                  ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:          fmul    s2, s29, v10.s[4]
// CHECK-ERROR:                                 ^

//----------------------------------------------------------------------
// Floating Point  multiply extended (scalar, by element)
//----------------------------------------------------------------------
      // mismatched and invalid vector types
      fmulx    d0, d1, v1.b[0]
      fmulx    h0, h1, v1.d[0]
      // invalid lane
      fmulx    d2, d29, v10.d[3]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fmulx    d0, d1, v1.b[0]
// CHECK-ERROR:                              ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fmulx    h0, h1, v1.d[0]
// CHECK-ERROR:                   ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:          fmulx    d2, d29, v10.d[3]
// CHECK-ERROR:                                  ^

//----------------------------------------------------------------------
// Floating Point fused multiply-add (scalar, by element)
//----------------------------------------------------------------------
      // mismatched and invalid vector types
      fmla    b0, b1, v1.b[0]
      fmla    d30, s11, v1.d[1]
      // invalid lane
      fmla    s16, s22, v16.s[5]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fmla    b0, b1, v1.b[0]
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fmla    d30, s11, v1.d[1]
// CHECK-ERROR:                       ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:          fmla    s16, s22, v16.s[5]
// CHECK-ERROR:                                  ^

//----------------------------------------------------------------------
// Floating Point fused multiply-subtract (scalar, by element)
//----------------------------------------------------------------------
    // mismatched and invalid vector types
    fmls    s29, h10, v28.s[1]
    fmls    h7, h17, v26.s[2]
    // invalid lane
    fmls    d16, d22, v16.d[-1]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fmls    s29, h10, v28.s[1]
// CHECK-ERROR:                       ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          fmls    h7, h17, v26.s[2]
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: vector lane must be an integer in range [0, 1]
// CHECK-ERROR:          fmls    d16, d22, v16.d[-1]
// CHECK-ERROR:                                  ^

//----------------------------------------------------------------------
// Scalar Signed saturating doubling multiply-add long
// (scalar, by element)
//----------------------------------------------------------------------
    // mismatched and invalid vector types
    sqdmlal s0, h0, v0.s[0]
    sqdmlal s8, s9, v14.s[1]
    // invalid lane
    sqdmlal d4, s5, v1.s[5]
    // invalid vector index
    sqdmlal s0, h0, v17.h[0]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          sqdmlal s0, h0, v0.s[0]
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          sqdmlal s8, s9, v14.s[1]
// CHECK-ERROR:                  ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:          sqdmlal d4, s5, v1.s[5]
// CHECK-ERROR:                               ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          sqdmlal s0, h0, v17.h[0]
// CHECK-ERROR:                           ^

//----------------------------------------------------------------------
// Scalar Signed saturating doubling multiply-subtract long
// (scalar, by element)
//----------------------------------------------------------------------
    // mismatched and invalid vector types
    sqdmlsl s1, h1, v1.d[0]
    sqdmlsl d1, h1, v13.s[0]
    // invalid lane
    sqdmlsl d1, s1, v13.s[4]
    // invalid vector index
    sqdmlsl s1, h1, v20.h[7]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          sqdmlsl s1, h1, v1.d[0]
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          sqdmlsl d1, h1, v13.s[0]
// CHECK-ERROR:                      ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:          sqdmlsl d1, s1, v13.s[4]
// CHECK-ERROR:                                ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          sqdmlsl s1, h1, v20.h[7]
// CHECK-ERROR:                           ^

//----------------------------------------------------------------------
// Scalar Signed saturating doubling multiply long (scalar, by element)
//----------------------------------------------------------------------
    // mismatched and invalid vector types
    // invalid lane
    // invalid vector index
    // mismatched and invalid vector types
    sqdmull s1, h1, v1.s[1]
    sqdmull s1, s1, v4.s[0]
    // invalid lane
    sqdmull s12, h17, v9.h[9]
    // invalid vector index
    sqdmull s1, h1, v16.h[5]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          sqdmull s1, h1, v1.s[1]
// CHECK-ERROR:                             ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          sqdmull s1, s1, v4.s[0]
// CHECK-ERROR:                  ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:          sqdmull s12, h17, v9.h[9]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          sqdmull s1, h1, v16.h[5]
// CHECK-ERROR:                           ^

//----------------------------------------------------------------------
// Scalar Signed saturating doubling multiply returning
// high half (scalar, by element)
//----------------------------------------------------------------------
    // mismatched and invalid vector types
    sqdmulh h0, s1, v0.h[0]
    sqdmulh s25, s26, v27.h[3]
    // invalid lane
    sqdmulh s25, s26, v27.s[4]
    // invalid vector index
    sqdmulh s0, h1, v30.h[0]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          sqdmulh h0, s1, v0.h[0]
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          sqdmulh s25, s26, v27.h[3]
// CHECK-ERROR:                  ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:          sqdmulh s25, s26, v27.s[4]
// CHECK-ERROR:                                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          sqdmulh s0, h1, v30.h[0]
// CHECK-ERROR:                      ^

//----------------------------------------------------------------------
// Scalar Signed saturating rounding doubling multiply
// returning high half (scalar, by element)
//----------------------------------------------------------------------
    // mismatched and invalid vector types
    sqrdmulh h31, h30, v14.s[2]
    sqrdmulh s5, h6, v7.s[2]
    // invalid lane
    sqrdmulh h31, h30, v14.h[9]
    // invalid vector index
    sqrdmulh h31, h30, v20.h[4]

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          sqrdmulh h31, h30, v14.s[2]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          sqrdmulh s5, h6, v7.s[2]
// CHECK-ERROR:                       ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:          sqrdmulh h31, h30, v14.h[9]
// CHECK-ERROR:                                 ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          sqrdmulh h31, h30, v20.h[4]
// CHECK-ERROR:                              ^

//----------------------------------------------------------------------
// Scalar Duplicate element (scalar)
//----------------------------------------------------------------------
      // mismatched and invalid vector types
      dup b0, v1.d[0]
      dup h0, v31.b[8]
      dup s0, v2.h[4]
      dup d0, v17.s[3]
      // invalid  lane
      dup d0, v17.d[4]
      dup s0, v1.s[7]
      dup h0, v31.h[16]
      dup b1, v3.b[16]
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          dup b0, v1.d[0]
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          dup h0, v31.b[8]
// CHECK-ERROR:                      ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          dup s0, v2.h[4]
// CHECK-ERROR:                     ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:          dup d0, v17.s[3]
// CHECK-ERROR:                      ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:          dup d0, v17.d[4]
// CHECK-ERROR:                        ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:          dup s0, v1.s[7]
// CHECK-ERROR:                       ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:          dup h0, v31.h[16]
// CHECK-ERROR:                        ^
// CHECK-ERROR: vector lane must be an integer in range
// CHECK-ERROR:          dup b1, v3.b[16]
// CHECK-ERROR:                       ^

//----------------------------------------------------------------------
// Table look up
//----------------------------------------------------------------------

        tbl v0.8b, {v1.8b}, v2.8b
        tbl v0.8b, {v1.8b, v2.8b}, v2.8b
        tbl v0.8b, {v1.8b, v2.8b, v3.8b}, v2.8b
        tbl v0.8b, {v1.8b, v2.8b, v3.8b, v4.8b}, v2.8b
        tbl v0.8b, {v1.16b, v2.16b, v3.16b, v4.16b, v5.16b}, v2.8b

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        tbl v0.8b, {v1.8b}, v2.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        tbl v0.8b, {v1.8b, v2.8b}, v2.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        tbl v0.8b, {v1.8b, v2.8b, v3.8b}, v2.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        tbl v0.8b, {v1.8b, v2.8b, v3.8b, v4.8b}, v2.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid number of vectors
// CHECK-ERROR:        tbl v0.8b, {v1.16b, v2.16b, v3.16b, v4.16b, v5.16b}, v2.8b
// CHECK-ERROR:                                                    ^

        tbx v0.8b, {v1.8b}, v2.8b
        tbx v0.8b, {v1.8b, v2.8b}, v2.8b
        tbx v0.8b, {v1.8b, v2.8b, v3.8b}, v2.8b
        tbx v0.8b, {v1.8b, v2.8b, v3.8b, v4.8b}, v2.8b
        tbx v0.8b, {v1.16b, v2.16b, v3.16b, v4.16b, v5.16b}, v2.8b

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        tbx v0.8b, {v1.8b}, v2.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        tbx v0.8b, {v1.8b, v2.8b}, v2.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        tbx v0.8b, {v1.8b, v2.8b, v3.8b}, v2.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        tbx v0.8b, {v1.8b, v2.8b, v3.8b, v4.8b}, v2.8b
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid number of vectors
// CHECK-ERROR:        tbx v0.8b, {v1.16b, v2.16b, v3.16b, v4.16b, v5.16b}, v2.8b
// CHECK-ERROR:                                                    ^

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Lower Precision Narrow, Rounding To
// Odd
//----------------------------------------------------------------------

    fcvtxn s0, s1

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtxn s0, s1
// CHECK-ERROR:                   ^

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Signed Integer, Rounding To Nearest
// With Ties To Away
//----------------------------------------------------------------------

    fcvtas s0, d0
    fcvtas d0, s0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtas s0, d0
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtas d0, s0
// CHECK-ERROR:                   ^

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Unsigned Integer, Rounding To
// Nearest With Ties To Away
//----------------------------------------------------------------------

    fcvtau s0, d0
    fcvtau d0, s0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtau s0, d0
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtau d0, s0
// CHECK-ERROR:                   ^

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Signed Integer, Rounding Toward
// Minus Infinity
//----------------------------------------------------------------------

    fcvtms s0, d0
    fcvtms d0, s0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtms s0, d0
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtms d0, s0
// CHECK-ERROR:                   ^

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Unsigned Integer, Rounding Toward
// Minus Infinity
//----------------------------------------------------------------------

    fcvtmu s0, d0
    fcvtmu d0, s0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtmu s0, d0
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtmu d0, s0
// CHECK-ERROR:                   ^

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Signed Integer, Rounding To Nearest
// With Ties To Even
//----------------------------------------------------------------------

    fcvtns s0, d0
    fcvtns d0, s0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtns s0, d0
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtns d0, s0
// CHECK-ERROR:                   ^

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Unsigned Integer, Rounding To
// Nearest With Ties To Even
//----------------------------------------------------------------------

    fcvtnu s0, d0
    fcvtnu d0, s0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtnu s0, d0
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtnu d0, s0
// CHECK-ERROR:                   ^

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Signed Integer, Rounding Toward
// Positive Infinity
//----------------------------------------------------------------------

    fcvtps s0, d0
    fcvtps d0, s0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtps s0, d0
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtps d0, s0
// CHECK-ERROR:                   ^

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Unsigned Integer, Rounding Toward
// Positive Infinity
//----------------------------------------------------------------------

    fcvtpu s0, d0
    fcvtpu d0, s0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtpu s0, d0
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtpu d0, s0
// CHECK-ERROR:                   ^

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Signed Integer, Rounding Toward Zero
//----------------------------------------------------------------------

    fcvtzs s0, d0
    fcvtzs d0, s0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtzs s0, d0
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtzs d0, s0
// CHECK-ERROR:                   ^

//----------------------------------------------------------------------
// Scalar Floating-point Convert To Unsigned Integer, Rounding Toward 
// Zero
//----------------------------------------------------------------------

    fcvtzu s0, d0
    fcvtzu d0, s0

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtzu s0, d0
// CHECK-ERROR:                   ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fcvtzu d0, s0
// CHECK-ERROR:                   ^

//----------------------------------------------------------------------
// Scalar Floating-point Absolute Difference
//----------------------------------------------------------------------


    fabd s29, d24, s20
    fabd d29, s24, d20

// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fabd s29, d24, s20
// CHECK-ERROR:                  ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:        fabd d29, s24, d20
// CHECK-ERROR:                  ^
