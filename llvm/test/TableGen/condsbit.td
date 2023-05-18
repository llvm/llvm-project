// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// Check that !cond works well with bit conditional values.

class A<bit b = true> {
  int a = !cond(b: 5, true : 6);
  bit c = !cond(b: false, true : true);
  bits<1> d = !cond(b: 0, true : 1);
}

// CHECK: def X
// CHECK:   a = 6
// CHECK:   c = 1
// CHECK:   d = { 1 }

// CHECK: def Y
// CHECK:   a = 5
// CHECK:   c = 0
// CHECK:   d = { 0 }

def X : A<false>;
def Y : A;
