// RUN: llvm-mc -triple aarch64 -show-encoding < %s | FileCheck %s

mrs x0, FGWTE3_EL3
// CHECK: mrs x0, FGWTE3_EL3                  // encoding: [0xa0,0x11,0x3e,0xd5]
msr FGWTE3_EL3, x0
// CHECK: msr FGWTE3_EL3, x0                  // encoding: [0xa0,0x11,0x1e,0xd5]
