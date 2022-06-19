// RUN: llvm-mc -triple aarch64 -show-encoding -mattr=+v8r -o - %s | FileCheck %s
.text
dfb

// CHECK:      .text
// CHECK-NEXT: dfb     // encoding: [0x9f,0x3c,0x03,0xd5]
