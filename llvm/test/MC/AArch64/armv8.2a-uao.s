// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.2a < %s 2> %t | FileCheck %s
// RUN: FileCheck --check-prefix=CHECK-ERROR %s < %t

  msr uao, #0
  msr uao, #1
// CHECK: msr     UAO, #0                 // encoding: [0x7f,0x40,0x00,0xd5]
// CHECK: msr     UAO, #1                 // encoding: [0x7f,0x41,0x00,0xd5]

  msr uao, #2
// CHECK-ERROR: error: immediate must be an integer in range [0, 1].
// CHECK-ERROR:   msr uao, #2
// CHECK-ERROR:            ^

  msr uao, x1
  mrs x2, uao
// CHECK: msr     UAO, x1                 // encoding: [0x81,0x42,0x18,0xd5]
// CHECK: mrs     x2, UAO                 // encoding: [0x82,0x42,0x38,0xd5]
