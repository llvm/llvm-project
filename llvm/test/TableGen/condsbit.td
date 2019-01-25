// check that !cond works well with bit conditional values
// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak
// CHECK: a = 6
// CHECK: a = 5

class A<bit b = 1> {
  bit true = 1;
  int a = !cond(b: 5, true : 6);
  bit c = !cond(b: 0, true : 1);
  bits<1> d = !cond(b: 0, true : 1);
}

def X : A<0>;
def Y : A;
