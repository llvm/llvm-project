// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding < %s | FileCheck %s

  csdb
// CHECK: csdb   // encoding: [0x9f,0x22,0x03,0xd5]
