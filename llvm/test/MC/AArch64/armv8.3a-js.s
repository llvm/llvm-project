// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+v8.3a -o - %s 2>&1 | \
// RUN: FileCheck %s

// RUN: llvm-mc -triple aarch64-none-linux-gnu -show-encoding -mattr=+jsconv -o - %s 2>&1 | \
// RUN: FileCheck %s

// RUN: not llvm-mc -triple aarch64-none-linux-gnu %s 2>&1 | \
// RUN: FileCheck --check-prefix=CHECK-JS %s

// RUN: not llvm-mc -triple aarch64-none-linux-gnu -mattr=+jsconv,-fp-armv8 -o - %s 2>&1 |\
// RUN: FileCheck --check-prefix=CHECK-REQ %s

  fjcvtzs w0, d0
// CHECK: fjcvtzs w0, d0    // encoding: [0x00,0x00,0x7e,0x1e]

// CHECK-JS: error: instruction requires: jsconv

// NOJS: error: instruction requires: jsconv

// CHECK-REQ: error: instruction requires: fp-armv8 jsconv
