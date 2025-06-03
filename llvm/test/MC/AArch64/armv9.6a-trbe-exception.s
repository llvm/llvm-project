// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding      < %s 2> %t | FileCheck %s

  msr trbsr_el12, x0
  msr trbsr_el2, x0
  msr trbsr_el3, x0
// CHECK:     msr TRBSR_EL12, x0          // encoding: [0x60,0x9b,0x1d,0xd5]
// CHECK:     msr TRBSR_EL2, x0           // encoding: [0x60,0x9b,0x1c,0xd5]
// CHECK:     msr TRBSR_EL3, x0           // encoding: [0x60,0x9b,0x1e,0xd5]

  mrs x0, trbsr_el12
  mrs x0, trbsr_el2
  mrs x0, trbsr_el3
// CHECK:    mrs x0, TRBSR_EL12          // encoding: [0x60,0x9b,0x3d,0xd5]
// CHECK:    mrs x0, TRBSR_EL2           // encoding: [0x60,0x9b,0x3c,0xd5]
// CHECK:    mrs x0, TRBSR_EL3           // encoding: [0x60,0x9b,0x3e,0xd5]
