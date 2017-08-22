// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.3a < %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-REQ < %t %s
// RUN: not llvm-mc -triple aarch64-none-linux-gnu -mattr=+v8.3a,-fp-armv8 < %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-NOFP < %t %s

  fjcvtzs w0, d0
// CHECK: fjcvtzs w0, d0    // encoding: [0x00,0x00,0x7e,0x1e]
// CHECK-REQ: error: instruction requires: armv8.3a
// CHECK-NOFP: error: instruction requires: fp-armv8
