// RUN: llvm-tblgen --print-detailed-records %s | FileCheck %s -DFILE=%{s:basename}

class A<int a> {
  int Num = a;
}

// Verify that the location of the anonymous record instantiated
// for A<10> and A<11> is correct. It should show the line where the
// anonymous record was instantiated and not the line where the class
// was defined.
def y {
  // CHECK: anonymous_0 |[[FILE]]:[[@LINE+1]]|
  int x = A<10>.Num;
  // CHECK: anonymous_1 |[[FILE]]:[[@LINE+1]]|
  int y = A<11>.Num;
}
