// RUN: llvm-mc -triple aarch64 -show-encoding < %s | FileCheck %s

mrs x0, VDISR_EL3
// CHECK: mrs x0, VDISR_EL3                  // encoding: [0x20,0xc1,0x3e,0xd5]

msr VDISR_EL3, x0
// CHECK: msr VDISR_EL3, x0                  // encoding: [0x20,0xc1,0x1e,0xd5]

mrs x0, VSESR_EL3
// CHECK: mrs x0, VSESR_EL3                  // encoding: [0x60,0x52,0x3e,0xd5]

msr VSESR_EL3, x0
// CHECK: msr VSESR_EL3, x0                  // encoding: [0x60,0x52,0x1e,0xd5]
