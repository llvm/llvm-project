// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding < %s | FileCheck %s

csdb
ssbb
pssbb

// CHECK: csdb   // encoding: [0x9f,0x22,0x03,0xd5]
// CHECK: ssbb   // encoding: [0x9f,0x30,0x03,0xd5]
// CHECK: pssbb  // encoding: [0x9f,0x34,0x03,0xd5]
