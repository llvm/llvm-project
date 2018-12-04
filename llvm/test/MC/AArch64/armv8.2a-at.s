// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.2a < %s | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=-v8.2a < %s 2>&1 | FileCheck %s --check-prefix=ERROR

  at s1e1rp, x1
  at s1e1wp, x2
// CHECK: at      s1e1rp, x1              // encoding: [0x01,0x79,0x08,0xd5]
// CHECK: at      s1e1wp, x2              // encoding: [0x22,0x79,0x08,0xd5]
// ERROR: error: AT S1E1RP requires ARMv8.2a
// ERROR: error: AT S1E1WP requires ARMv8.2a
