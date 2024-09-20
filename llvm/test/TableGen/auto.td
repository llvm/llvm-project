// RUN: llvm-tblgen %s | FileCheck %s --check-prefix=CHECK-PASS
// RUN: not llvm-tblgen -DBAD0 %s 2>&1 | FileCheck %s --check-prefix=CHECK-BAD0 -DFILE=%s
// RUN: not llvm-tblgen -DBAD1 %s 2>&1 | FileCheck %s --check-prefix=CHECK-BAD1 -DFILE=%s

// CHECK-PASS: int Arg = A:P;
class A<int P> {
  auto Arg = P;
}

def MyOp;

// CHECK-PASS-LABEL: class B
// CHECK-PASS:  int A = 10;
// CHECK-PASS:  string B = "x";
// CHECK-PASS:  list<int> C = [10, 10];
// CHECK-PASS{LITERAL}:  list<list<int>> D = [[10, 10], [11, 11]];
// CHECK-PASS:  list<A> F = [anonymous_0, anonymous_1];
// CHECK-PASS:  bits<4> G = { 0, 1, 0, 1 };
// CHECK-PASS:  code H = [{ printf(); }];
// CHECK-PASS:  bits<1> I = { 0 };
// CHECK-PASS:  dag K = (MyOp 10, 100);
// CHECK-PASS:  int L = 10;
class B {
  auto A = 10;
  auto B = "x";
  auto C = [10,10];
  auto D = [[10,10],[11,11]];
#ifdef BAD0
  // CHECK-BAD0: [[FILE]]:[[@LINE+1]]:12: error: unable to infer type
  auto E = [];
#endif  
  auto F = [A<10>, A<11>];
  auto G = {0,1,0,1};
  auto H = [{ printf(); }];
  // FIXME: This becomes `bits` and not `bit`.
  auto I = 0b0;
#ifdef BAD1
  // CHECK-BAD1: [[FILE]]:[[@LINE+1]]:12: error: unable to infer type
  auto J = ?;
#endif  
  auto K = (MyOp 10, 100);
  auto L = A<10>.Arg;
}

