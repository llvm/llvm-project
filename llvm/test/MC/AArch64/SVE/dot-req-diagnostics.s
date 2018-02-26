// RUN: not llvm-mc -triple aarch64-none-linux-gnu -mattr=+sve < %s 2>&1 | FileCheck --check-prefix=CHECK --check-prefix=CHECK-ERROR %s

foo:
// CHECK: error: sve predicate register without type specifier expected
  pbarb .req p1.b
// CHECK: error: sve predicate register without type specifier expected
  pbarh .req p1.h
// CHECK: error: sve predicate register without type specifier expected
  pbars .req p1.s
// CHECK: error: sve predicate register without type specifier expected
  pbard .req p1.d

// CHECK: error: sve vector register without type specifier expected
  zbarb .req z1.b
// CHECK: error: sve vector register without type specifier expected
  zbarh .req z1.h
// CHECK: error: sve vector register without type specifier expected
  zbars .req z1.s
// CHECK: error: sve vector register without type specifier expected
  zbard .req z1.d
