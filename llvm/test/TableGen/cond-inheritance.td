// Make sure !cond gets propagated across multiple layers of inheritance.
// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

class getInt<int c> {
  int ret = !cond(c: 0, 1 : 1);
}

class I1<int c> {
  int i = getInt<c>.ret;
}

class I2<int c> : I1<c>;

def DI1: I1<1>;
// CHECK: def DI1 {     // I1
// CHECK-NEXT: int i = 0;

// CHECK: def DI2 {     // I1 I2
// CHECK-NEXT: int i = 0;
def DI2: I2<1>;

