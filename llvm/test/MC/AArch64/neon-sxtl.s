// RUN: llvm-mc -triple=aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//------------------------------------------------------------------------------
// Signed integer lengthen (vector)
//------------------------------------------------------------------------------
        sxtl v0.8h, v1.8b
        sxtl v0.4s, v1.4h
        sxtl v0.2d, v1.2s

// CHECK: sshll v0.8h, v1.8b, #0        // encoding: [0x20,0xa4,0x08,0x0f]
// CHECK: sshll v0.4s, v1.4h, #0        // encoding: [0x20,0xa4,0x10,0x0f]
// CHECK: sshll v0.2d, v1.2s, #0        // encoding: [0x20,0xa4,0x20,0x0f]

//------------------------------------------------------------------------------
// Signed integer lengthen (vector, second part)
//------------------------------------------------------------------------------

        sxtl2 v0.8h, v1.16b
        sxtl2 v0.4s, v1.8h
        sxtl2 v0.2d, v1.4s

// CHECK: sshll2 v0.8h, v1.16b, #0        // encoding: [0x20,0xa4,0x08,0x4f]
// CHECK: sshll2 v0.4s, v1.8h, #0        // encoding: [0x20,0xa4,0x10,0x4f]
// CHECK: sshll2 v0.2d, v1.4s, #0        // encoding: [0x20,0xa4,0x20,0x4f]
