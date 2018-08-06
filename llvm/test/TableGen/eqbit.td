// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK-LABEL: def X {
//       CHECK:   int a = 6;
//       CHECK:   int c = 5;
//       CHECK: }

// CHECK-LABEL: def Y {
//       CHECK:   int a = 5;
//       CHECK:   int c = 6;
//       CHECK: }

class A<bit b = 1> {
  int a = !if(!eq(b, 1), 5, 6);
  int c = !if(!ne(b, 1), 5, 6);
}

def X : A<0>;
def Y : A;
