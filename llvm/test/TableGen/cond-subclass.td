// Check that !cond with operands of different subtypes can
// initialize a supertype variable.
// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

class E<int dummy> {}
class E1<int dummy> : E<dummy> {}
class E2<int dummy> : E<dummy> {}

class EX<int cc, E1 b, E2 c> {
  E x = !cond(cc: b, 1 : c);
}

def E1d : E1<0>;
def E2d : E2<0>;

def EXd1 : EX<1, E1d, E2d>;
def EXd2 : EX<0, E1d, E2d>;

// CHECK: def EXd1 {
// CHECK:   E x = E1d;
// CHECK: }
//
// CHECK: def EXd2 {
// CHECK:   E x = E2d;
// CHECK: }

