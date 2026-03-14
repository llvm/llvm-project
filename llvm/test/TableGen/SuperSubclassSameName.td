// RUN: llvm-tblgen < %s
// XFAIL: vg_leak
// Test for template arguments that have the same name as superclass template
// arguments.


class Arg { int a; }
def TheArg : Arg { let a = 1; }


class Super<Arg F> {
  int X = F.a;
}
class Sub<Arg F> : Super<F>;
def inst : Sub<TheArg>;


class Super2<int F> {
  int X = F;
}
class Sub2<int F> : Super2<F>;
