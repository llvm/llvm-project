// RUN: not llvm-mc -triple aarch64-none-linux-gnu -mattr=+v8.1a -show-encoding < %s 2> %t | FileCheck %s
// RUN: FileCheck --check-prefix=CHECK-ERROR %s < %t
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -mattr=+v8r -show-encoding < %s 2> %t | FileCheck %s
// RUN: FileCheck --check-prefix=CHECK-ERROR %s < %t

  .text

  msr pan, #0
// CHECK:  msr PAN, #0           // encoding: [0x9f,0x40,0x00,0xd5]
  msr pan, #1
// CHECK:  msr PAN, #1           // encoding: [0x9f,0x41,0x00,0xd5]
  msr pan, x5
// CHECK:  msr PAN, x5           // encoding: [0x65,0x42,0x18,0xd5]
  mrs x13, pan
// CHECK:  mrs x13, PAN          // encoding: [0x6d,0x42,0x38,0xd5]

  msr pan, #-1
  msr pan, #2
  msr pan, w0
  mrs w0, pan
// CHECK-ERROR: error: immediate must be an integer in range [0, 1].
// CHECK-ERROR:   msr pan, #-1
// CHECK-ERROR:            ^
// CHECK-ERROR: error: immediate must be an integer in range [0, 1].
// CHECK-ERROR:   msr pan, #2
// CHECK-ERROR:            ^
// CHECK-ERROR: error: immediate must be an integer in range [0, 1].
// CHECK-ERROR:   msr pan, w0
// CHECK-ERROR:            ^
// CHECK-ERROR: error: invalid operand for instruction
// CHECK-ERROR:   mrs w0, pan
// CHECK-ERROR:       ^
