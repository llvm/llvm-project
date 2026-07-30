# RUN: llvm-mc -triple i386-unknown-unknown %s | FileCheck %s

// We used to incorrectly parse a line with only a # in it

.zero 42
#
.ifndef FOO
.zero 2
.else
.endif
.zero 24

// CHECK: .zero 42
// CHECK-NEXT: .zero 2
// CHECK-NEXT: .zero 24
