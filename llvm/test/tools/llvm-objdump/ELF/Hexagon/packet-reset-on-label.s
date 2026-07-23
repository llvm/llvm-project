// RUN: llvm-mc -triple=hexagon -mcpu=hexagonv75 -filetype=obj %s \
// RUN:   | llvm-objdump -d - \
// RUN:   | FileCheck %s

foo:
  { nop }
  /// a nop without end-of-packet bits set to simulate data that is
  /// not a proper packet end.
  .long 0x7f004000
bar:
  { nop
    nop
  }

// CHECK-LABEL: <foo>:
// CHECK: { nop }
// CHECK-NEXT: { nop

/// The instruction starting after <bar> should start in a new packet.
// CHECK-LABEL: <bar>:
// CHECK: { nop
// CHECK-NEXT: nop }

