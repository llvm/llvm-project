// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

class A<int k, bits<2> x = 1> {
  int K = k;
  bits<2> Bits = x;
}

// CHECK: def a1
// CHECK: Bits = { 0, 1 }
def a1 : A<12>;

// CHECK: def a2
// CHECK: Bits = { 1, 0 }
def a2 : A<13, 2>;

// Here was the bug: X.Bits would get resolved to the default a1.Bits while
// resolving the first template argument. When the second template argument
// was processed, X would be set correctly, but Bits retained the default
// value.
class B<int k, A x = a1> {
  A X = x;
  bits<2> Bits = X.Bits;
}

// CHECK: def b1
// CHECK: Bits = { 0, 1 }
def b1 : B<27>;

// CHECK: def b2
// CHECK: Bits = { 1, 0 }
def b2 : B<28, a2>;

class C<A x = a1> {
  bits<2> Bits = x.Bits;
}

// CHECK: def c1
// CHECK: Bits = { 0, 1 }
def c1 : C;

// CHECK: def c2
// CHECK: Bits = { 1, 0 }
def c2 : C<a2>;
