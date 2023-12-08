// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak
// CHECK: a = 6
// CHECK: a = 5

class A<bit b = 1> {
  int a = !if(b, 5, 6);
  bit c = !if(b, 0, 1);
  bits<1> d = !if(b, 0, 1);
}

def X : A<0>;
def Y : A;
