// RUN: llvm-mc -triple=arm64 -mattr=+neon -mattr=+crypto -show-encoding < %s | FileCheck %s
// RUN: not llvm-mc -triple=arm64 -mattr=+neon -show-encoding < %s 2>&1 | FileCheck -check-prefix=CHECK-NO-CRYPTO-ARM64 %s

// Check that the assembler can handle the documented syntax for AArch64

//------------------------------------------------------------------------------
// Instructions for crypto
//------------------------------------------------------------------------------

        aese v0.16b, v1.16b
        aesd v0.16b, v1.16b
        aesmc v0.16b, v1.16b
        aesimc v0.16b, v1.16b

// CHECK-NO-CRYPTO: error: instruction requires a CPU feature not currently enabled
// CHECK-NO-CRYPTO-ARM64: error: instruction requires: aes
// CHECK: aese v0.16b, v1.16b          // encoding: [0x20,0x48,0x28,0x4e]
// CHECK: aesd v0.16b, v1.16b          // encoding: [0x20,0x58,0x28,0x4e]
// CHECK: aesmc v0.16b, v1.16b         // encoding: [0x20,0x68,0x28,0x4e]
// CHECK: aesimc v0.16b, v1.16b        // encoding: [0x20,0x78,0x28,0x4e]

        sha1h s0, s1
        sha1su1 v0.4s, v1.4s
        sha256su0 v0.4s, v1.4s

// CHECK: sha1h s0, s1                  // encoding: [0x20,0x08,0x28,0x5e]
// CHECK: sha1su1 v0.4s, v1.4s          // encoding: [0x20,0x18,0x28,0x5e]
// CHECK: sha256su0 v0.4s, v1.4s        // encoding: [0x20,0x28,0x28,0x5e]

        sha1c q0, s1, v2.4s
        sha1p q0, s1, v2.4s
        sha1m q0, s1, v2.4s
        sha1su0 v0.4s, v1.4s, v2.4s
        sha256h q0, q1, v2.4s
        sha256h2 q0, q1, v2.4s
        sha256su1 v0.4s, v1.4s, v2.4s

// CHECK: sha1c q0, s1, v2.4s           // encoding: [0x20,0x00,0x02,0x5e]
// CHECK: sha1p q0, s1, v2.4s           // encoding: [0x20,0x10,0x02,0x5e]
// CHECK: sha1m q0, s1, v2.4s           // encoding: [0x20,0x20,0x02,0x5e]
// CHECK: sha1su0 v0.4s, v1.4s, v2.4s   // encoding: [0x20,0x30,0x02,0x5e]
// CHECK: sha256h q0, q1, v2.4s         // encoding: [0x20,0x40,0x02,0x5e]
// CHECK: sha256h2 q0, q1, v2.4s        // encoding: [0x20,0x50,0x02,0x5e]
// CHECK: sha256su1 v0.4s, v1.4s, v2.4s // encoding: [0x20,0x60,0x02,0x5e]
