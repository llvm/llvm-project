// RUN:     llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+predctrl < %s      | FileCheck %s
// RUN:     llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.5a    < %s      | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=-predctrl < %s 2>&1 | FileCheck %s --check-prefix=NOPREDCTRL

cfp rctx, x0
dvp rctx, x1
cpp rctx, x2

// CHECK: cfp rctx, x0      // encoding: [0x80,0x73,0x0b,0xd5]
// CHECK: dvp rctx, x1      // encoding: [0xa1,0x73,0x0b,0xd5]
// CHECK: cpp rctx, x2      // encoding: [0xe2,0x73,0x0b,0xd5]

// NOPREDCTRL: CFPRCTX requires predctrl
// NOPREDCTRL-NEXT: cfp
// NOPREDCTRL: DVPRCTX requires predctrl
// NOPREDCTRL-NEXT: dvp
// NOPREDCTRL: CPPRCTX requires predctrl
// NOPREDCTRL-NEXT: cpp
