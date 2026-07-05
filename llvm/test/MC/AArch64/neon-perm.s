// RUN: llvm-mc -triple=arm64 -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//------------------------------------------------------------------------------
// Instructions for permute
//------------------------------------------------------------------------------

        uzp1 v0.8b, v1.8b, v2.8b
        uzp1 v0.16b, v1.16b, v2.16b
        uzp1 v0.4h, v1.4h, v2.4h
        uzp1 v0.8h, v1.8h, v2.8h
        uzp1 v0.2s, v1.2s, v2.2s
        uzp1 v0.4s, v1.4s, v2.4s
        uzp1 v0.2d, v1.2d, v2.2d

// CHECK: uzp1	v0.8b, v1.8b, v2.8b     // encoding: [0x20,0x18,0x02,0x0e]
// CHECK: uzp1	v0.16b, v1.16b, v2.16b  // encoding: [0x20,0x18,0x02,0x4e]
// CHECK: uzp1	v0.4h, v1.4h, v2.4h     // encoding: [0x20,0x18,0x42,0x0e]
// CHECK: uzp1	v0.8h, v1.8h, v2.8h     // encoding: [0x20,0x18,0x42,0x4e]
// CHECK: uzp1	v0.2s, v1.2s, v2.2s     // encoding: [0x20,0x18,0x82,0x0e]
// CHECK: uzp1	v0.4s, v1.4s, v2.4s     // encoding: [0x20,0x18,0x82,0x4e]
// CHECK: uzp1	v0.2d, v1.2d, v2.2d     // encoding: [0x20,0x18,0xc2,0x4e]

        trn1 v0.8b, v1.8b, v2.8b
        trn1 v0.16b, v1.16b, v2.16b
        trn1 v0.4h, v1.4h, v2.4h
        trn1 v0.8h, v1.8h, v2.8h
        trn1 v0.2s, v1.2s, v2.2s
        trn1 v0.4s, v1.4s, v2.4s
        trn1 v0.2d, v1.2d, v2.2d

// CHECK: trn1	v0.8b, v1.8b, v2.8b     // encoding: [0x20,0x28,0x02,0x0e]
// CHECK: trn1	v0.16b, v1.16b, v2.16b  // encoding: [0x20,0x28,0x02,0x4e]
// CHECK: trn1	v0.4h, v1.4h, v2.4h     // encoding: [0x20,0x28,0x42,0x0e]
// CHECK: trn1	v0.8h, v1.8h, v2.8h     // encoding: [0x20,0x28,0x42,0x4e]
// CHECK: trn1	v0.2s, v1.2s, v2.2s     // encoding: [0x20,0x28,0x82,0x0e]
// CHECK: trn1	v0.4s, v1.4s, v2.4s     // encoding: [0x20,0x28,0x82,0x4e]
// CHECK: trn1	v0.2d, v1.2d, v2.2d     // encoding: [0x20,0x28,0xc2,0x4e]

        zip1 v0.8b, v1.8b, v2.8b
        zip1 v0.16b, v1.16b, v2.16b
        zip1 v0.4h, v1.4h, v2.4h
        zip1 v0.8h, v1.8h, v2.8h
        zip1 v0.2s, v1.2s, v2.2s
        zip1 v0.4s, v1.4s, v2.4s
        zip1 v0.2d, v1.2d, v2.2d

// CHECK: zip1	v0.8b, v1.8b, v2.8b     // encoding: [0x20,0x38,0x02,0x0e]
// CHECK: zip1	v0.16b, v1.16b, v2.16b  // encoding: [0x20,0x38,0x02,0x4e]
// CHECK: zip1	v0.4h, v1.4h, v2.4h     // encoding: [0x20,0x38,0x42,0x0e]
// CHECK: zip1	v0.8h, v1.8h, v2.8h     // encoding: [0x20,0x38,0x42,0x4e]
// CHECK: zip1	v0.2s, v1.2s, v2.2s     // encoding: [0x20,0x38,0x82,0x0e]
// CHECK: zip1	v0.4s, v1.4s, v2.4s     // encoding: [0x20,0x38,0x82,0x4e]
// CHECK: zip1	v0.2d, v1.2d, v2.2d     // encoding: [0x20,0x38,0xc2,0x4e]

        uzp2 v0.8b, v1.8b, v2.8b
        uzp2 v0.16b, v1.16b, v2.16b
        uzp2 v0.4h, v1.4h, v2.4h
        uzp2 v0.8h, v1.8h, v2.8h
        uzp2 v0.2s, v1.2s, v2.2s
        uzp2 v0.4s, v1.4s, v2.4s
        uzp2 v0.2d, v1.2d, v2.2d

// CHECK: uzp2	v0.8b, v1.8b, v2.8b     // encoding: [0x20,0x58,0x02,0x0e]
// CHECK: uzp2	v0.16b, v1.16b, v2.16b  // encoding: [0x20,0x58,0x02,0x4e]
// CHECK: uzp2	v0.4h, v1.4h, v2.4h     // encoding: [0x20,0x58,0x42,0x0e]
// CHECK: uzp2	v0.8h, v1.8h, v2.8h     // encoding: [0x20,0x58,0x42,0x4e]
// CHECK: uzp2	v0.2s, v1.2s, v2.2s     // encoding: [0x20,0x58,0x82,0x0e]
// CHECK: uzp2	v0.4s, v1.4s, v2.4s     // encoding: [0x20,0x58,0x82,0x4e]
// CHECK: uzp2	v0.2d, v1.2d, v2.2d     // encoding: [0x20,0x58,0xc2,0x4e]

        trn2 v0.8b, v1.8b, v2.8b
        trn2 v0.16b, v1.16b, v2.16b
        trn2 v0.4h, v1.4h, v2.4h
        trn2 v0.8h, v1.8h, v2.8h
        trn2 v0.2s, v1.2s, v2.2s
        trn2 v0.4s, v1.4s, v2.4s
        trn2 v0.2d, v1.2d, v2.2d

// CHECK: trn2	v0.8b, v1.8b, v2.8b     // encoding: [0x20,0x68,0x02,0x0e]
// CHECK: trn2	v0.16b, v1.16b, v2.16b  // encoding: [0x20,0x68,0x02,0x4e]
// CHECK: trn2	v0.4h, v1.4h, v2.4h     // encoding: [0x20,0x68,0x42,0x0e]
// CHECK: trn2	v0.8h, v1.8h, v2.8h     // encoding: [0x20,0x68,0x42,0x4e]
// CHECK: trn2	v0.2s, v1.2s, v2.2s     // encoding: [0x20,0x68,0x82,0x0e]
// CHECK: trn2	v0.4s, v1.4s, v2.4s     // encoding: [0x20,0x68,0x82,0x4e]
// CHECK: trn2	v0.2d, v1.2d, v2.2d     // encoding: [0x20,0x68,0xc2,0x4e]

        zip2 v0.8b, v1.8b, v2.8b
        zip2 v0.16b, v1.16b, v2.16b
        zip2 v0.4h, v1.4h, v2.4h
        zip2 v0.8h, v1.8h, v2.8h
        zip2 v0.2s, v1.2s, v2.2s
        zip2 v0.4s, v1.4s, v2.4s
        zip2 v0.2d, v1.2d, v2.2d

// CHECK: zip2	v0.8b, v1.8b, v2.8b     // encoding: [0x20,0x78,0x02,0x0e]
// CHECK: zip2	v0.16b, v1.16b, v2.16b  // encoding: [0x20,0x78,0x02,0x4e]
// CHECK: zip2	v0.4h, v1.4h, v2.4h     // encoding: [0x20,0x78,0x42,0x0e]
// CHECK: zip2	v0.8h, v1.8h, v2.8h     // encoding: [0x20,0x78,0x42,0x4e]
// CHECK: zip2	v0.2s, v1.2s, v2.2s     // encoding: [0x20,0x78,0x82,0x0e]
// CHECK: zip2	v0.4s, v1.4s, v2.4s     // encoding: [0x20,0x78,0x82,0x4e]
// CHECK: zip2	v0.2d, v1.2d, v2.2d     // encoding: [0x20,0x78,0xc2,0x4e]
