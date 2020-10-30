// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// Check that !cond works well with bit conditional values.

// CHECK: a = 6
// CHECK: a = 5

class A<bit b = 1> {
  bit True = 1;
  int a = !cond(b: 5, True : 6);
  bit c = !cond(b: 0, True : 1);
  bits<1> d = !cond(b: 0, True : 1);
}

def X : A<0>;
def Y : A;
