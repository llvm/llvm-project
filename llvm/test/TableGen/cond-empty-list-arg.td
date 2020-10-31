// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// Check that !cond works with an empty list value.

class C<bit cond> {
  bit True = 1;
  list<int> X = !cond(cond: [1, 2, 3], True : []);
  list<int> Y = !cond(cond: [], True : [4, 5, 6]);
}

// CHECK: def rec1
// CHECK:   X = [];
// CHECK:   Y = [4, 5, 6];

def rec1 : C<0>;

// CHECK: def rec2
// CHECK:   X = [1, 2, 3];
// CHECK:   Y = [];

def rec2 : C<1>;
