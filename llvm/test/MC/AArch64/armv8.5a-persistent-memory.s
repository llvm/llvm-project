// RUN:     llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+ccdp  < %s      | FileCheck %s
// RUN:     llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.5a < %s      | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=-ccdp  < %s 2>&1 | FileCheck %s --check-prefix=NOCCDP

dc cvadp, x7
// CHECK:  dc cvadp, x7   // encoding: [0x27,0x7d,0x0b,0xd5]
// NOCCDP: error: DC CVADP requires: ccdp
