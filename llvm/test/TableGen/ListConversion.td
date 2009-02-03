// RUN: tblgen %s
class A;
class B : A;

def b : B;

def {
  list<B> X = [b];
  list<A> Y = X;
}
