// RUN:     llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+specres2 < %s      | FileCheck %s
// RUN:     llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.9a    < %s      | FileCheck %s
// RUN:     llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v9.4a    < %s      | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=-specres2 < %s 2>&1 | FileCheck %s --check-prefix=NOSPECRES2

cosp rctx, x0
sys #3, c7, c3, #6, x0

// CHECK: cosp rctx, x0          // encoding: [0xc0,0x73,0x0b,0xd5]
// CHECK: cosp rctx, x0          // encoding: [0xc0,0x73,0x0b,0xd5]

// NOSPECRES2: COSP requires: predres2
// NOSPECRES2-NEXT: cosp
