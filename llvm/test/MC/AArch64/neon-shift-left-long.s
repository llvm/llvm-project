// RUN: llvm-mc -triple aarch64-none-linux-gnu -mattr=+neon -show-encoding < %s | FileCheck %s

// Check that the assembler can handle the documented syntax for AArch64

//------------------------------------------------------------------------------
// Integer shift left long (Signed)
//------------------------------------------------------------------------------
         sshll v0.8h, v1.8b, #3
         sshll v0.4s, v1.4h, #3
         sshll v0.2d, v1.2s, #3
         sshll2 v0.8h, v1.16b, #3
         sshll2 v0.4s, v1.8h, #3
         sshll2 v0.2d, v1.4s, #3

// CHECK: sshll v0.8h, v1.8b, #3         // encoding: [0x20,0xa4,0x0b,0x0f]
// CHECK: sshll v0.4s, v1.4h, #3         // encoding: [0x20,0xa4,0x13,0x0f]
// CHECK: sshll v0.2d, v1.2s, #3         // encoding: [0x20,0xa4,0x23,0x0f]
// CHECK: sshll2 v0.8h, v1.16b, #3       // encoding: [0x20,0xa4,0x0b,0x4f]
// CHECK: sshll2 v0.4s, v1.8h, #3        // encoding: [0x20,0xa4,0x13,0x4f]
// CHECK: sshll2 v0.2d, v1.4s, #3        // encoding: [0x20,0xa4,0x23,0x4f]

//------------------------------------------------------------------------------
// Integer shift left long (Unsigned)
//------------------------------------------------------------------------------
         ushll v0.8h, v1.8b, #3
         ushll v0.4s, v1.4h, #3
         ushll v0.2d, v1.2s, #3
         ushll2 v0.8h, v1.16b, #3
         ushll2 v0.4s, v1.8h, #3
         ushll2 v0.2d, v1.4s, #3

// CHECK: ushll v0.8h, v1.8b, #3         // encoding: [0x20,0xa4,0x0b,0x2f]
// CHECK: ushll v0.4s, v1.4h, #3         // encoding: [0x20,0xa4,0x13,0x2f]
// CHECK: ushll v0.2d, v1.2s, #3         // encoding: [0x20,0xa4,0x23,0x2f]
// CHECK: ushll2 v0.8h, v1.16b, #3       // encoding: [0x20,0xa4,0x0b,0x6f]
// CHECK: ushll2 v0.4s, v1.8h, #3        // encoding: [0x20,0xa4,0x13,0x6f]
// CHECK: ushll2 v0.2d, v1.4s, #3        // encoding: [0x20,0xa4,0x23,0x6f]
