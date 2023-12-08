// RUN: llvm-tblgen %s
// XFAIL: vg_leak
class A;
class B : A;

def b : B;

def {
  list<B> X = [b];
  list<A> Y = X;
}
