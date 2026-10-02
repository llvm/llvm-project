// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding < %s | FileCheck %s

  mrs x2, HCRX_EL2
// CHECK: mrs x2, HCRX_EL2              // encoding: [0x42,0x12,0x3c,0xd5]

  msr HCRX_EL2, x3
// CHECK: msr HCRX_EL2, x3              // encoding: [0x43,0x12,0x1c,0xd5]
