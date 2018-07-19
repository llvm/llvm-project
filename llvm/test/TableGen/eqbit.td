// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak
// CHECK: a = 6
// CHECK: a = 5

class A<bit b = 1> {
  int a = !if(!eq(b, 1), 5, 6);
}

def X : A<0>;
def Y : A;
