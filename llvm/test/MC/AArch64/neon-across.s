// RUN: llvm-mc -triple=arm64 -mattr=+neon,+fullfp16 -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//------------------------------------------------------------------------------
// Instructions across vector registers
//------------------------------------------------------------------------------

        saddlv h0, v1.8b
        saddlv h0, v1.16b
        saddlv s0, v1.4h
        saddlv s0, v1.8h
        saddlv d0, v1.4s

// CHECK: saddlv	h0, v1.8b               // encoding: [0x20,0x38,0x30,0x0e]
// CHECK: saddlv	h0, v1.16b              // encoding: [0x20,0x38,0x30,0x4e]
// CHECK: saddlv	s0, v1.4h               // encoding: [0x20,0x38,0x70,0x0e]
// CHECK: saddlv	s0, v1.8h               // encoding: [0x20,0x38,0x70,0x4e]
// CHECK: saddlv	d0, v1.4s               // encoding: [0x20,0x38,0xb0,0x4e]

        uaddlv h0, v1.8b
        uaddlv h0, v1.16b
        uaddlv s0, v1.4h
        uaddlv s0, v1.8h
        uaddlv d0, v1.4s

// CHECK: uaddlv	h0, v1.8b               // encoding: [0x20,0x38,0x30,0x2e]
// CHECK: uaddlv	h0, v1.16b              // encoding: [0x20,0x38,0x30,0x6e]
// CHECK: uaddlv	s0, v1.4h               // encoding: [0x20,0x38,0x70,0x2e]
// CHECK: uaddlv	s0, v1.8h               // encoding: [0x20,0x38,0x70,0x6e]
// CHECK: uaddlv	d0, v1.4s               // encoding: [0x20,0x38,0xb0,0x6e]

        smaxv b0, v1.8b
        smaxv b0, v1.16b
        smaxv h0, v1.4h
        smaxv h0, v1.8h
        smaxv s0, v1.4s

// CHECK: smaxv	b0, v1.8b               // encoding: [0x20,0xa8,0x30,0x0e]
// CHECK: smaxv	b0, v1.16b              // encoding: [0x20,0xa8,0x30,0x4e]
// CHECK: smaxv	h0, v1.4h               // encoding: [0x20,0xa8,0x70,0x0e]
// CHECK: smaxv	h0, v1.8h               // encoding: [0x20,0xa8,0x70,0x4e]
// CHECK: smaxv	s0, v1.4s               // encoding: [0x20,0xa8,0xb0,0x4e]

        sminv b0, v1.8b
        sminv b0, v1.16b
        sminv h0, v1.4h
        sminv h0, v1.8h
        sminv s0, v1.4s

// CHECK: sminv	b0, v1.8b               // encoding: [0x20,0xa8,0x31,0x0e]
// CHECK: sminv	b0, v1.16b              // encoding: [0x20,0xa8,0x31,0x4e]
// CHECK: sminv	h0, v1.4h               // encoding: [0x20,0xa8,0x71,0x0e]
// CHECK: sminv	h0, v1.8h               // encoding: [0x20,0xa8,0x71,0x4e]
// CHECK: sminv	s0, v1.4s               // encoding: [0x20,0xa8,0xb1,0x4e]

        umaxv b0, v1.8b
        umaxv b0, v1.16b
        umaxv h0, v1.4h
        umaxv h0, v1.8h
        umaxv s0, v1.4s

// CHECK: umaxv	b0, v1.8b               // encoding: [0x20,0xa8,0x30,0x2e]
// CHECK: umaxv	b0, v1.16b              // encoding: [0x20,0xa8,0x30,0x6e]
// CHECK: umaxv	h0, v1.4h               // encoding: [0x20,0xa8,0x70,0x2e]
// CHECK: umaxv	h0, v1.8h               // encoding: [0x20,0xa8,0x70,0x6e]
// CHECK: umaxv	s0, v1.4s               // encoding: [0x20,0xa8,0xb0,0x6e]

        uminv b0, v1.8b
        uminv b0, v1.16b
        uminv h0, v1.4h
        uminv h0, v1.8h
        uminv s0, v1.4s

// CHECK: uminv	b0, v1.8b               // encoding: [0x20,0xa8,0x31,0x2e]
// CHECK: uminv	b0, v1.16b              // encoding: [0x20,0xa8,0x31,0x6e]
// CHECK: uminv	h0, v1.4h               // encoding: [0x20,0xa8,0x71,0x2e]
// CHECK: uminv	h0, v1.8h               // encoding: [0x20,0xa8,0x71,0x6e]
// CHECK: uminv	s0, v1.4s               // encoding: [0x20,0xa8,0xb1,0x6e]

        addv b0, v1.8b
        addv b0, v1.16b
        addv h0, v1.4h
        addv h0, v1.8h
        addv s0, v1.4s

// CHECK: addv	b0, v1.8b               // encoding: [0x20,0xb8,0x31,0x0e]
// CHECK: addv	b0, v1.16b              // encoding: [0x20,0xb8,0x31,0x4e]
// CHECK: addv	h0, v1.4h               // encoding: [0x20,0xb8,0x71,0x0e]
// CHECK: addv	h0, v1.8h               // encoding: [0x20,0xb8,0x71,0x4e]
// CHECK: addv	s0, v1.4s               // encoding: [0x20,0xb8,0xb1,0x4e]

        fmaxnmv h0, v1.4h
        fminnmv h0, v1.4h
        fmaxv h0, v1.4h
        fminv h0, v1.4h
        fmaxnmv h0, v1.8h
        fminnmv h0, v1.8h
        fmaxv h0, v1.8h
        fminv h0, v1.8h
        fmaxnmv s0, v1.4s
        fminnmv s0, v1.4s
        fmaxv s0, v1.4s
        fminv s0, v1.4s

// CHECK: fmaxnmv h0, v1.4h               // encoding: [0x20,0xc8,0x30,0x0e]
// CHECK: fminnmv h0, v1.4h               // encoding: [0x20,0xc8,0xb0,0x0e]
// CHECK: fmaxv   h0, v1.4h               // encoding: [0x20,0xf8,0x30,0x0e]
// CHECK: fminv   h0, v1.4h               // encoding: [0x20,0xf8,0xb0,0x0e]
// CHECK: fmaxnmv h0, v1.8h               // encoding: [0x20,0xc8,0x30,0x4e]
// CHECK: fminnmv h0, v1.8h               // encoding: [0x20,0xc8,0xb0,0x4e]
// CHECK: fmaxv   h0, v1.8h               // encoding: [0x20,0xf8,0x30,0x4e]
// CHECK: fminv   h0, v1.8h               // encoding: [0x20,0xf8,0xb0,0x4e]
// CHECK: fmaxnmv	s0, v1.4s               // encoding: [0x20,0xc8,0x30,0x6e]
// CHECK: fminnmv	s0, v1.4s               // encoding: [0x20,0xc8,0xb0,0x6e]
// CHECK: fmaxv	s0, v1.4s               // encoding: [0x20,0xf8,0x30,0x6e]
// CHECK: fminv	s0, v1.4s               // encoding: [0x20,0xf8,0xb0,0x6e]
