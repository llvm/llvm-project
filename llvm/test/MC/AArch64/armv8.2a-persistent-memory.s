// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.2a -o - %s | FileCheck %s
// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+ccpp -o - %s | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=-v8.2a -o - %s 2>&1 | FileCheck %s --check-prefix=ERROR

  dc cvap, x7
// CHECK: dc cvap, x7   // encoding: [0x27,0x7c,0x0b,0xd5]
// ERROR: error: DC CVAP requires ccpp
