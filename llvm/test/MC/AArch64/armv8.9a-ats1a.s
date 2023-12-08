// RUN: llvm-mc -triple aarch64 -show-encoding %s | FileCheck %s

at s1e1a, x1
// CHECK: at s1e1a, x1                        // encoding: [0x41,0x79,0x08,0xd5]

at s1e2a, x1
// CHECK: at s1e2a, x1                        // encoding: [0x41,0x79,0x0c,0xd5]

at s1e3a, x1
// CHECK: at s1e3a, x1                        // encoding: [0x41,0x79,0x0e,0xd5]
