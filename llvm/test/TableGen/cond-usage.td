// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// Check that !cond picks the first true value
// CHECK:       class A
// CHECK-NEXT:  string S = !cond(!eq(A:x, 10): "ten", !eq(A:x, 11): "eleven", !eq(A:x, 10): "TEN", !gt(A:x, 9): "MoreThanNine", 1: "unknown"); 
// CHECK: B1
// CHECK-NEXT: string S = "unknown"
// CHECK: B10
// CHECK-NEXT: string S = "ten";
// CHECK: def B11
// CHECK-NEXT: string S = "eleven";
// CHECK: def B12
// CHECK-NEXT:  string S = "MoreThanNine";
// CHECK: def B9
// CHECK-NEXT: string S = "unknown"

class A<int x> {
  string S = !cond(!eq(x,10) : "ten",
                   !eq(x,11) : "eleven",
                   !eq(x,10) : "TEN",
                   !gt(x,9) : "MoreThanNine",
                   !eq(1,1) : "unknown");
}
def B1  : A<1>;
def B9  : A<9>;
def B10 : A<10>;
def B11 : A<11>;
def B12 : A<12>;
