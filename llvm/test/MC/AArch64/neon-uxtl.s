// RUN: llvm-mc -triple=aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//------------------------------------------------------------------------------
// Unsigned integer lengthen (vector)
//------------------------------------------------------------------------------
        uxtl v0.8h, v1.8b
        uxtl v0.4s, v1.4h
        uxtl v0.2d, v1.2s

// CHECK: ushll v0.8h, v1.8b, #0        // encoding: [0x20,0xa4,0x08,0x2f]
// CHECK: ushll v0.4s, v1.4h, #0        // encoding: [0x20,0xa4,0x10,0x2f]
// CHECK: ushll v0.2d, v1.2s, #0        // encoding: [0x20,0xa4,0x20,0x2f]

//------------------------------------------------------------------------------
// Unsigned integer lengthen (vector, second part)
//------------------------------------------------------------------------------

        uxtl2 v0.8h, v1.16b
        uxtl2 v0.4s, v1.8h
        uxtl2 v0.2d, v1.4s

// CHECK: ushll2 v0.8h, v1.16b, #0       // encoding: [0x20,0xa4,0x08,0x6f]
// CHECK: ushll2 v0.4s, v1.8h, #0        // encoding: [0x20,0xa4,0x10,0x6f]
// CHECK: ushll2 v0.2d, v1.4s, #0        // encoding: [0x20,0xa4,0x20,0x6f]
