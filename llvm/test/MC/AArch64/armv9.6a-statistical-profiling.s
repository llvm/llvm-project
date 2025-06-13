// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding     < %s 2> %t | FileCheck %s

  msr pmbmar_el1, x0
  msr pmbsr_el12, x0
  msr pmbsr_el2, x0
  msr pmbsr_el3, x0
// CHECK:     msr PMBMAR_EL1, x0          // encoding: [0xa0,0x9a,0x18,0xd5]
// CHECK:     msr PMBSR_EL12, x0          // encoding: [0x60,0x9a,0x1d,0xd5]
// CHECK:     msr PMBSR_EL2, x0           // encoding: [0x60,0x9a,0x1c,0xd5]
// CHECK:     msr PMBSR_EL3, x0           // encoding: [0x60,0x9a,0x1e,0xd5]

  mrs x0, pmbmar_el1
  mrs x0, pmbsr_el12
  mrs x0, pmbsr_el2
  mrs x0, pmbsr_el3
// CHECK:    mrs x0, PMBMAR_EL1          // encoding: [0xa0,0x9a,0x38,0xd5]
// CHECK:    mrs x0, PMBSR_EL12          // encoding: [0x60,0x9a,0x3d,0xd5]
// CHECK:    mrs x0, PMBSR_EL2           // encoding: [0x60,0x9a,0x3c,0xd5]
// CHECK:    mrs x0, PMBSR_EL3           // encoding: [0x60,0x9a,0x3e,0xd5]
