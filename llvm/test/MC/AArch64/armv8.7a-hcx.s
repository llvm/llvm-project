// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+hcx < %s 2>%t | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.7a < %s 2>%t | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NO-HCX-ERR %s < %t

  mrs x2, HCRX_EL2
// CHECK: mrs x2, HCRX_EL2              // encoding: [0x42,0x12,0x3c,0xd5]
// CHECK-NO-HCX-ERR: [[@LINE-2]]:11: error: expected readable system register

  msr HCRX_EL2, x3
// CHECK: msr HCRX_EL2, x3              // encoding: [0x43,0x12,0x1c,0xd5]
// CHECK-NO-HCX-ERR: [[@LINE-2]]:7: error: expected writable system register
