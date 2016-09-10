// RUN: llvm-mc -arch=hexagon -filetype=obj -o - %s | llvm-objdump -d - | FileCheck %s
// REQUIRES: asserts
  .text
// CHECK: { r7 = #-1; r7 = #-1 }
  .long 0x3a373a27
// CHECK: { if (!p0.new) r7 = #0; if (p0.new) r7 = #0 }
  .long 0x3a573a47
