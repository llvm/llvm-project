// RUN: llvm-mc -triple=aarch64-none-linux-gnu -mattr=+crypto -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//------------------------------------------------------------------------------
// Instructions with 3 different vector data types
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Long
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// Long - Variant 1
//------------------------------------------------------------------------------

        saddl v0.8h, v1.8b, v2.8b
        saddl v0.4s, v1.4h, v2.4h
        saddl v0.2d, v1.2s, v2.2s

// CHECK: saddl	v0.8h, v1.8b, v2.8b     // encoding: [0x20,0x00,0x22,0x0e]
// CHECK: saddl	v0.4s, v1.4h, v2.4h     // encoding: [0x20,0x00,0x62,0x0e]
// CHECK: saddl	v0.2d, v1.2s, v2.2s     // encoding: [0x20,0x00,0xa2,0x0e]

        saddl2 v0.4s, v1.8h, v2.8h
        saddl2 v0.8h, v1.16b, v2.16b
        saddl2 v0.2d, v1.4s, v2.4s

// CHECK: saddl2	v0.4s, v1.8h, v2.8h     // encoding: [0x20,0x00,0x62,0x4e]
// CHECK: saddl2	v0.8h, v1.16b, v2.16b   // encoding: [0x20,0x00,0x22,0x4e]
// CHECK: saddl2	v0.2d, v1.4s, v2.4s     // encoding: [0x20,0x00,0xa2,0x4e]

        uaddl v0.8h, v1.8b, v2.8b
        uaddl v0.4s, v1.4h, v2.4h
        uaddl v0.2d, v1.2s, v2.2s

// CHECK: uaddl	v0.8h, v1.8b, v2.8b     // encoding: [0x20,0x00,0x22,0x2e]
// CHECK: uaddl	v0.4s, v1.4h, v2.4h     // encoding: [0x20,0x00,0x62,0x2e]
// CHECK: uaddl	v0.2d, v1.2s, v2.2s     // encoding: [0x20,0x00,0xa2,0x2e]

        uaddl2 v0.8h, v1.16b, v2.16b
        uaddl2 v0.4s, v1.8h, v2.8h
        uaddl2 v0.2d, v1.4s, v2.4s

// CHECK: uaddl2	v0.8h, v1.16b, v2.16b   // encoding: [0x20,0x00,0x22,0x6e]
// CHECK: uaddl2	v0.4s, v1.8h, v2.8h     // encoding: [0x20,0x00,0x62,0x6e]
// CHECK: uaddl2	v0.2d, v1.4s, v2.4s     // encoding: [0x20,0x00,0xa2,0x6e]

        ssubl v0.8h, v1.8b, v2.8b
        ssubl v0.4s, v1.4h, v2.4h
        ssubl v0.2d, v1.2s, v2.2s

// CHECK: ssubl	v0.8h, v1.8b, v2.8b     // encoding: [0x20,0x20,0x22,0x0e]
// CHECK: ssubl	v0.4s, v1.4h, v2.4h     // encoding: [0x20,0x20,0x62,0x0e]
// CHECK: ssubl	v0.2d, v1.2s, v2.2s     // encoding: [0x20,0x20,0xa2,0x0e]

        ssubl2 v0.8h, v1.16b, v2.16b
        ssubl2 v0.4s, v1.8h, v2.8h
        ssubl2 v0.2d, v1.4s, v2.4s

// CHECK: ssubl2	v0.8h, v1.16b, v2.16b   // encoding: [0x20,0x20,0x22,0x4e]
// CHECK: ssubl2	v0.4s, v1.8h, v2.8h     // encoding: [0x20,0x20,0x62,0x4e]
// CHECK: ssubl2	v0.2d, v1.4s, v2.4s     // encoding: [0x20,0x20,0xa2,0x4e]

        usubl v0.8h, v1.8b, v2.8b
        usubl v0.4s, v1.4h, v2.4h
        usubl v0.2d, v1.2s, v2.2s

// CHECK: usubl	v0.8h, v1.8b, v2.8b     // encoding: [0x20,0x20,0x22,0x2e]
// CHECK: usubl	v0.4s, v1.4h, v2.4h     // encoding: [0x20,0x20,0x62,0x2e]
// CHECK: usubl	v0.2d, v1.2s, v2.2s     // encoding: [0x20,0x20,0xa2,0x2e]

        usubl2 v0.8h, v1.16b, v2.16b
        usubl2 v0.4s, v1.8h, v2.8h
        usubl2 v0.2d, v1.4s, v2.4s

// CHECK: usubl2	v0.8h, v1.16b, v2.16b   // encoding: [0x20,0x20,0x22,0x6e]
// CHECK: usubl2	v0.4s, v1.8h, v2.8h     // encoding: [0x20,0x20,0x62,0x6e]
// CHECK: usubl2	v0.2d, v1.4s, v2.4s     // encoding: [0x20,0x20,0xa2,0x6e]

        sabal v0.8h, v1.8b, v2.8b
        sabal v0.4s, v1.4h, v2.4h
        sabal v0.2d, v1.2s, v2.2s

// CHECK: sabal	v0.8h, v1.8b, v2.8b     // encoding: [0x20,0x50,0x22,0x0e]
// CHECK: sabal	v0.4s, v1.4h, v2.4h     // encoding: [0x20,0x50,0x62,0x0e]
// CHECK: sabal	v0.2d, v1.2s, v2.2s     // encoding: [0x20,0x50,0xa2,0x0e]

        sabal2 v0.8h, v1.16b, v2.16b
        sabal2 v0.4s, v1.8h, v2.8h
        sabal2 v0.2d, v1.4s, v2.4s

// CHECK: sabal2	v0.8h, v1.16b, v2.16b   // encoding: [0x20,0x50,0x22,0x4e]
// CHECK: sabal2	v0.4s, v1.8h, v2.8h     // encoding: [0x20,0x50,0x62,0x4e]
// CHECK: sabal2	v0.2d, v1.4s, v2.4s     // encoding: [0x20,0x50,0xa2,0x4e]

        uabal v0.8h, v1.8b, v2.8b
        uabal v0.4s, v1.4h, v2.4h
        uabal v0.2d, v1.2s, v2.2s

// CHECK: uabal	v0.8h, v1.8b, v2.8b     // encoding: [0x20,0x50,0x22,0x2e]
// CHECK: uabal	v0.4s, v1.4h, v2.4h     // encoding: [0x20,0x50,0x62,0x2e]
// CHECK: uabal	v0.2d, v1.2s, v2.2s     // encoding: [0x20,0x50,0xa2,0x2e]

        uabal2 v0.8h, v1.16b, v2.16b
        uabal2 v0.4s, v1.8h, v2.8h
        uabal2 v0.2d, v1.4s, v2.4s

// CHECK: uabal2	v0.8h, v1.16b, v2.16b   // encoding: [0x20,0x50,0x22,0x6e]
// CHECK: uabal2	v0.4s, v1.8h, v2.8h     // encoding: [0x20,0x50,0x62,0x6e]
// CHECK: uabal2	v0.2d, v1.4s, v2.4s     // encoding: [0x20,0x50,0xa2,0x6e]

        sabdl v0.8h, v1.8b, v2.8b
        sabdl v0.4s, v1.4h, v2.4h
        sabdl v0.2d, v1.2s, v2.2s

// CHECK: sabdl	v0.8h, v1.8b, v2.8b     // encoding: [0x20,0x70,0x22,0x0e]
// CHECK: sabdl	v0.4s, v1.4h, v2.4h     // encoding: [0x20,0x70,0x62,0x0e]
// CHECK: sabdl	v0.2d, v1.2s, v2.2s     // encoding: [0x20,0x70,0xa2,0x0e]

        sabdl2 v0.8h, v1.16b, v2.16b
        sabdl2 v0.4s, v1.8h, v2.8h
        sabdl2 v0.2d, v1.4s, v2.4s

// CHECK: sabdl2	v0.8h, v1.16b, v2.16b   // encoding: [0x20,0x70,0x22,0x4e]
// CHECK: sabdl2	v0.4s, v1.8h, v2.8h     // encoding: [0x20,0x70,0x62,0x4e]
// CHECK: sabdl2	v0.2d, v1.4s, v2.4s     // encoding: [0x20,0x70,0xa2,0x4e]

        uabdl v0.8h, v1.8b, v2.8b
        uabdl v0.4s, v1.4h, v2.4h
        uabdl v0.2d, v1.2s, v2.2s

// CHECK: uabdl	v0.8h, v1.8b, v2.8b     // encoding: [0x20,0x70,0x22,0x2e]
// CHECK: uabdl	v0.4s, v1.4h, v2.4h     // encoding: [0x20,0x70,0x62,0x2e]
// CHECK: uabdl	v0.2d, v1.2s, v2.2s     // encoding: [0x20,0x70,0xa2,0x2e]

        uabdl2 v0.8h, v1.16b, v2.16b
        uabdl2 v0.4s, v1.8h, v2.8h
        uabdl2 v0.2d, v1.4s, v2.4s

// CHECK: uabdl2	v0.8h, v1.16b, v2.16b   // encoding: [0x20,0x70,0x22,0x6e]
// CHECK: uabdl2	v0.4s, v1.8h, v2.8h     // encoding: [0x20,0x70,0x62,0x6e]
// CHECK: uabdl2	v0.2d, v1.4s, v2.4s     // encoding: [0x20,0x70,0xa2,0x6e]

        smlal v0.8h, v1.8b, v2.8b
        smlal v0.4s, v1.4h, v2.4h
        smlal v0.2d, v1.2s, v2.2s

// CHECK: smlal	v0.8h, v1.8b, v2.8b     // encoding: [0x20,0x80,0x22,0x0e]
// CHECK: smlal	v0.4s, v1.4h, v2.4h     // encoding: [0x20,0x80,0x62,0x0e]
// CHECK: smlal	v0.2d, v1.2s, v2.2s     // encoding: [0x20,0x80,0xa2,0x0e]

        smlal2 v0.8h, v1.16b, v2.16b
        smlal2 v0.4s, v1.8h, v2.8h
        smlal2 v0.2d, v1.4s, v2.4s

// CHECK: smlal2	v0.8h, v1.16b, v2.16b   // encoding: [0x20,0x80,0x22,0x4e]
// CHECK: smlal2	v0.4s, v1.8h, v2.8h     // encoding: [0x20,0x80,0x62,0x4e]
// CHECK: smlal2	v0.2d, v1.4s, v2.4s     // encoding: [0x20,0x80,0xa2,0x4e]

        umlal v0.8h, v1.8b, v2.8b
        umlal v0.4s, v1.4h, v2.4h
        umlal v0.2d, v1.2s, v2.2s

// CHECK: umlal	v0.8h, v1.8b, v2.8b     // encoding: [0x20,0x80,0x22,0x2e]
// CHECK: umlal	v0.4s, v1.4h, v2.4h     // encoding: [0x20,0x80,0x62,0x2e]
// CHECK: umlal	v0.2d, v1.2s, v2.2s     // encoding: [0x20,0x80,0xa2,0x2e]

        umlal2 v0.8h, v1.16b, v2.16b
        umlal2 v0.4s, v1.8h, v2.8h
        umlal2 v0.2d, v1.4s, v2.4s

// CHECK: umlal2	v0.8h, v1.16b, v2.16b   // encoding: [0x20,0x80,0x22,0x6e]
// CHECK: umlal2	v0.4s, v1.8h, v2.8h     // encoding: [0x20,0x80,0x62,0x6e]
// CHECK: umlal2	v0.2d, v1.4s, v2.4s     // encoding: [0x20,0x80,0xa2,0x6e]

        smlsl v0.8h, v1.8b, v2.8b
        smlsl v0.4s, v1.4h, v2.4h
        smlsl v0.2d, v1.2s, v2.2s

// CHECK: smlsl	v0.8h, v1.8b, v2.8b     // encoding: [0x20,0xa0,0x22,0x0e]
// CHECK: smlsl	v0.4s, v1.4h, v2.4h     // encoding: [0x20,0xa0,0x62,0x0e]
// CHECK: smlsl	v0.2d, v1.2s, v2.2s     // encoding: [0x20,0xa0,0xa2,0x0e]

        smlsl2 v0.8h, v1.16b, v2.16b
        smlsl2 v0.4s, v1.8h, v2.8h
        smlsl2 v0.2d, v1.4s, v2.4s

// CHECK: smlsl2	v0.8h, v1.16b, v2.16b   // encoding: [0x20,0xa0,0x22,0x4e]
// CHECK: smlsl2	v0.4s, v1.8h, v2.8h     // encoding: [0x20,0xa0,0x62,0x4e]
// CHECK: smlsl2	v0.2d, v1.4s, v2.4s     // encoding: [0x20,0xa0,0xa2,0x4e]

        umlsl v0.8h, v1.8b, v2.8b
        umlsl v0.4s, v1.4h, v2.4h
        umlsl v0.2d, v1.2s, v2.2s

// CHECK: umlsl	v0.8h, v1.8b, v2.8b     // encoding: [0x20,0xa0,0x22,0x2e]
// CHECK: umlsl	v0.4s, v1.4h, v2.4h     // encoding: [0x20,0xa0,0x62,0x2e]
// CHECK: umlsl	v0.2d, v1.2s, v2.2s     // encoding: [0x20,0xa0,0xa2,0x2e]

        umlsl2 v0.8h, v1.16b, v2.16b
        umlsl2 v0.4s, v1.8h, v2.8h
        umlsl2 v0.2d, v1.4s, v2.4s

// CHECK: umlsl2	v0.8h, v1.16b, v2.16b   // encoding: [0x20,0xa0,0x22,0x6e]
// CHECK: umlsl2	v0.4s, v1.8h, v2.8h     // encoding: [0x20,0xa0,0x62,0x6e]
// CHECK: umlsl2	v0.2d, v1.4s, v2.4s     // encoding: [0x20,0xa0,0xa2,0x6e]

        smull v0.8h, v1.8b, v2.8b
        smull v0.4s, v1.4h, v2.4h
        smull v0.2d, v1.2s, v2.2s

// CHECK: smull	v0.8h, v1.8b, v2.8b     // encoding: [0x20,0xc0,0x22,0x0e]
// CHECK: smull	v0.4s, v1.4h, v2.4h     // encoding: [0x20,0xc0,0x62,0x0e]
// CHECK: smull	v0.2d, v1.2s, v2.2s     // encoding: [0x20,0xc0,0xa2,0x0e]

        smull2 v0.8h, v1.16b, v2.16b
        smull2 v0.4s, v1.8h, v2.8h
        smull2 v0.2d, v1.4s, v2.4s

// CHECK: smull2	v0.8h, v1.16b, v2.16b   // encoding: [0x20,0xc0,0x22,0x4e]
// CHECK: smull2	v0.4s, v1.8h, v2.8h     // encoding: [0x20,0xc0,0x62,0x4e]
// CHECK: smull2	v0.2d, v1.4s, v2.4s     // encoding: [0x20,0xc0,0xa2,0x4e]

        umull v0.8h, v1.8b, v2.8b
        umull v0.4s, v1.4h, v2.4h
        umull v0.2d, v1.2s, v2.2s

// CHECK: umull	v0.8h, v1.8b, v2.8b     // encoding: [0x20,0xc0,0x22,0x2e]
// CHECK: umull	v0.4s, v1.4h, v2.4h     // encoding: [0x20,0xc0,0x62,0x2e]
// CHECK: umull	v0.2d, v1.2s, v2.2s     // encoding: [0x20,0xc0,0xa2,0x2e]

        umull2 v0.8h, v1.16b, v2.16b
        umull2 v0.4s, v1.8h, v2.8h
        umull2 v0.2d, v1.4s, v2.4s

// CHECK: umull2	v0.8h, v1.16b, v2.16b   // encoding: [0x20,0xc0,0x22,0x6e]
// CHECK: umull2	v0.4s, v1.8h, v2.8h     // encoding: [0x20,0xc0,0x62,0x6e]
// CHECK: umull2	v0.2d, v1.4s, v2.4s     // encoding: [0x20,0xc0,0xa2,0x6e]

//------------------------------------------------------------------------------
// Long - Variant 2
//------------------------------------------------------------------------------

        sqdmlal v0.4s, v1.4h, v2.4h
        sqdmlal v0.2d, v1.2s, v2.2s

// CHECK: sqdmlal	v0.4s, v1.4h, v2.4h     // encoding: [0x20,0x90,0x62,0x0e]
// CHECK: sqdmlal	v0.2d, v1.2s, v2.2s     // encoding: [0x20,0x90,0xa2,0x0e]

        sqdmlal2 v0.4s, v1.8h, v2.8h
        sqdmlal2 v0.2d, v1.4s, v2.4s

// CHECK: sqdmlal2	v0.4s, v1.8h, v2.8h // encoding: [0x20,0x90,0x62,0x4e]
// CHECK: sqdmlal2	v0.2d, v1.4s, v2.4s // encoding: [0x20,0x90,0xa2,0x4e]

        sqdmlsl v0.4s, v1.4h, v2.4h
        sqdmlsl v0.2d, v1.2s, v2.2s

// CHECK: sqdmlsl	v0.4s, v1.4h, v2.4h     // encoding: [0x20,0xb0,0x62,0x0e]
// CHECK: sqdmlsl	v0.2d, v1.2s, v2.2s     // encoding: [0x20,0xb0,0xa2,0x0e]

        sqdmlsl2 v0.4s, v1.8h, v2.8h
        sqdmlsl2 v0.2d, v1.4s, v2.4s

// CHECK: sqdmlsl2	v0.4s, v1.8h, v2.8h // encoding: [0x20,0xb0,0x62,0x4e]
// CHECK: sqdmlsl2	v0.2d, v1.4s, v2.4s // encoding: [0x20,0xb0,0xa2,0x4e]

        sqdmull v0.4s, v1.4h, v2.4h
        sqdmull v0.2d, v1.2s, v2.2s

// CHECK: sqdmull	v0.4s, v1.4h, v2.4h     // encoding: [0x20,0xd0,0x62,0x0e]
// CHECK: sqdmull	v0.2d, v1.2s, v2.2s     // encoding: [0x20,0xd0,0xa2,0x0e]

        sqdmull2 v0.4s, v1.8h, v2.8h
        sqdmull2 v0.2d, v1.4s, v2.4s

// CHECK: sqdmull2	v0.4s, v1.8h, v2.8h // encoding: [0x20,0xd0,0x62,0x4e]
// CHECK: sqdmull2	v0.2d, v1.4s, v2.4s // encoding: [0x20,0xd0,0xa2,0x4e]

//------------------------------------------------------------------------------
// Long - Variant 3
//------------------------------------------------------------------------------

        pmull v0.8h, v1.8b, v2.8b
        pmull v0.1q, v1.1d, v2.1d

// CHECK: pmull	v0.8h, v1.8b, v2.8b     // encoding: [0x20,0xe0,0x22,0x0e]
// CHECK: pmull	v0.1q, v1.1d, v2.1d     // encoding: [0x20,0xe0,0xe2,0x0e]

        pmull2 v0.8h, v1.16b, v2.16b
        pmull2 v0.1q, v1.2d, v2.2d

// CHECK: pmull2	v0.8h, v1.16b, v2.16b   // encoding: [0x20,0xe0,0x22,0x4e]
// CHECK: pmull2	v0.1q, v1.2d, v2.2d     // encoding: [0x20,0xe0,0xe2,0x4e]

//------------------------------------------------------------------------------
// Widen
//------------------------------------------------------------------------------

        saddw v0.8h, v1.8h, v2.8b
        saddw v0.4s, v1.4s, v2.4h
        saddw v0.2d, v1.2d, v2.2s

// CHECK: saddw	v0.8h, v1.8h, v2.8b     // encoding: [0x20,0x10,0x22,0x0e]
// CHECK: saddw	v0.4s, v1.4s, v2.4h     // encoding: [0x20,0x10,0x62,0x0e]
// CHECK: saddw	v0.2d, v1.2d, v2.2s     // encoding: [0x20,0x10,0xa2,0x0e]

        saddw2 v0.8h, v1.8h, v2.16b
        saddw2 v0.4s, v1.4s, v2.8h
        saddw2 v0.2d, v1.2d, v2.4s

// CHECK: saddw2	v0.8h, v1.8h, v2.16b    // encoding: [0x20,0x10,0x22,0x4e]
// CHECK: saddw2	v0.4s, v1.4s, v2.8h     // encoding: [0x20,0x10,0x62,0x4e]
// CHECK: saddw2	v0.2d, v1.2d, v2.4s     // encoding: [0x20,0x10,0xa2,0x4e]

        uaddw v0.8h, v1.8h, v2.8b
        uaddw v0.4s, v1.4s, v2.4h
        uaddw v0.2d, v1.2d, v2.2s

// CHECK: uaddw	v0.8h, v1.8h, v2.8b     // encoding: [0x20,0x10,0x22,0x2e]
// CHECK: uaddw	v0.4s, v1.4s, v2.4h     // encoding: [0x20,0x10,0x62,0x2e]
// CHECK: uaddw	v0.2d, v1.2d, v2.2s     // encoding: [0x20,0x10,0xa2,0x2e]

        uaddw2 v0.8h, v1.8h, v2.16b
        uaddw2 v0.4s, v1.4s, v2.8h
        uaddw2 v0.2d, v1.2d, v2.4s

// CHECK: uaddw2	v0.8h, v1.8h, v2.16b    // encoding: [0x20,0x10,0x22,0x6e]
// CHECK: uaddw2	v0.4s, v1.4s, v2.8h     // encoding: [0x20,0x10,0x62,0x6e]
// CHECK: uaddw2	v0.2d, v1.2d, v2.4s     // encoding: [0x20,0x10,0xa2,0x6e]

        ssubw v0.8h, v1.8h, v2.8b
        ssubw v0.4s, v1.4s, v2.4h
        ssubw v0.2d, v1.2d, v2.2s

// CHECK: ssubw	v0.8h, v1.8h, v2.8b     // encoding: [0x20,0x30,0x22,0x0e]
// CHECK: ssubw	v0.4s, v1.4s, v2.4h     // encoding: [0x20,0x30,0x62,0x0e]
// CHECK: ssubw	v0.2d, v1.2d, v2.2s     // encoding: [0x20,0x30,0xa2,0x0e]

        ssubw2 v0.8h, v1.8h, v2.16b
        ssubw2 v0.4s, v1.4s, v2.8h
        ssubw2 v0.2d, v1.2d, v2.4s

// CHECK: ssubw2	v0.8h, v1.8h, v2.16b    // encoding: [0x20,0x30,0x22,0x4e]
// CHECK: ssubw2	v0.4s, v1.4s, v2.8h     // encoding: [0x20,0x30,0x62,0x4e]
// CHECK: ssubw2	v0.2d, v1.2d, v2.4s     // encoding: [0x20,0x30,0xa2,0x4e]

        usubw v0.8h, v1.8h, v2.8b
        usubw v0.4s, v1.4s, v2.4h
        usubw v0.2d, v1.2d, v2.2s

// CHECK: usubw	v0.8h, v1.8h, v2.8b     // encoding: [0x20,0x30,0x22,0x2e]
// CHECK: usubw	v0.4s, v1.4s, v2.4h     // encoding: [0x20,0x30,0x62,0x2e]
// CHECK: usubw	v0.2d, v1.2d, v2.2s     // encoding: [0x20,0x30,0xa2,0x2e]

        usubw2 v0.8h, v1.8h, v2.16b
        usubw2 v0.4s, v1.4s, v2.8h
        usubw2 v0.2d, v1.2d, v2.4s

// CHECK: usubw2	v0.8h, v1.8h, v2.16b    // encoding: [0x20,0x30,0x22,0x6e]
// CHECK: usubw2	v0.4s, v1.4s, v2.8h     // encoding: [0x20,0x30,0x62,0x6e]
// CHECK: usubw2	v0.2d, v1.2d, v2.4s     // encoding: [0x20,0x30,0xa2,0x6e]

//------------------------------------------------------------------------------
// Narrow
//------------------------------------------------------------------------------

        addhn v0.8b, v1.8h, v2.8h
        addhn v0.4h, v1.4s, v2.4s
        addhn v0.2s, v1.2d, v2.2d

// CHECK: addhn	v0.8b, v1.8h, v2.8h     // encoding: [0x20,0x40,0x22,0x0e]
// CHECK: addhn	v0.4h, v1.4s, v2.4s     // encoding: [0x20,0x40,0x62,0x0e]
// CHECK: addhn	v0.2s, v1.2d, v2.2d     // encoding: [0x20,0x40,0xa2,0x0e]

        addhn2 v0.16b, v1.8h, v2.8h
        addhn2 v0.8h, v1.4s, v2.4s
        addhn2 v0.4s, v1.2d, v2.2d

// CHECK: addhn2	v0.16b, v1.8h, v2.8h    // encoding: [0x20,0x40,0x22,0x4e]
// CHECK: addhn2	v0.8h, v1.4s, v2.4s     // encoding: [0x20,0x40,0x62,0x4e]
// CHECK: addhn2	v0.4s, v1.2d, v2.2d     // encoding: [0x20,0x40,0xa2,0x4e]

        raddhn v0.8b, v1.8h, v2.8h
        raddhn v0.4h, v1.4s, v2.4s
        raddhn v0.2s, v1.2d, v2.2d

// CHECK: raddhn	v0.8b, v1.8h, v2.8h     // encoding: [0x20,0x40,0x22,0x2e]
// CHECK: raddhn	v0.4h, v1.4s, v2.4s     // encoding: [0x20,0x40,0x62,0x2e]
// CHECK: raddhn	v0.2s, v1.2d, v2.2d     // encoding: [0x20,0x40,0xa2,0x2e]

        raddhn2 v0.16b, v1.8h, v2.8h
        raddhn2 v0.8h, v1.4s, v2.4s
        raddhn2 v0.4s, v1.2d, v2.2d

// CHECK: raddhn2	v0.16b, v1.8h, v2.8h    // encoding: [0x20,0x40,0x22,0x6e]
// CHECK: raddhn2	v0.8h, v1.4s, v2.4s     // encoding: [0x20,0x40,0x62,0x6e]
// CHECK: raddhn2	v0.4s, v1.2d, v2.2d     // encoding: [0x20,0x40,0xa2,0x6e]

        rsubhn v0.8b, v1.8h, v2.8h
        rsubhn v0.4h, v1.4s, v2.4s
        rsubhn v0.2s, v1.2d, v2.2d

// CHECK: rsubhn	v0.8b, v1.8h, v2.8h     // encoding: [0x20,0x60,0x22,0x2e]
// CHECK: rsubhn	v0.4h, v1.4s, v2.4s     // encoding: [0x20,0x60,0x62,0x2e]
// CHECK: rsubhn	v0.2s, v1.2d, v2.2d     // encoding: [0x20,0x60,0xa2,0x2e]

        rsubhn2 v0.16b, v1.8h, v2.8h
        rsubhn2 v0.8h, v1.4s, v2.4s
        rsubhn2 v0.4s, v1.2d, v2.2d

// CHECK: rsubhn2	v0.16b, v1.8h, v2.8h    // encoding: [0x20,0x60,0x22,0x6e]
// CHECK: rsubhn2	v0.8h, v1.4s, v2.4s     // encoding: [0x20,0x60,0x62,0x6e]
// CHECK: rsubhn2	v0.4s, v1.2d, v2.2d     // encoding: [0x20,0x60,0xa2,0x6e]
