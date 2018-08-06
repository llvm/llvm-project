// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: --- Defs ---

// CHECK: def A0 {
// CHECK:   dag a = (ops A0);
// CHECK: }

// CHECK: def B0 {
// CHECK:   dag a = (ops);
// CHECK:   A b = B0;
// CHECK: }

// CHECK: def C0 {
// CHECK:   dag q = (ops C0);
// CHECK: }

// CHECK: def D0 {
// CHECK:   D d = D0;
// CHECK: }

// CHECK: def E0 {
// CHECK:   E e = E0;
// CHECK: }

// CHECK: def F0 {
// CHECK:   Fa as_a = F0;
// CHECK:   Fb as_b = F0;
// CHECK: }
// CHECK: def F0x {
// CHECK:   Fc as_c = F0;
// CHECK: }

def ops;

class A<dag d> {
  dag a = d;
}

// This type of self-reference is used in various places defining register
// classes.
def A0 : A<(ops A0)>;

class B<string self> {
  A b = !cast<A>(self);
}

// A stronger form of this type of self-reference is used at least in the
// SystemZ backend to define a record which is a ComplexPattern and an Operand
// at the same time.
def B0 : A<(ops)>, B<"B0">;

// Casting C0 to C by name here is tricky, because it happens while (or rather:
// before) adding C as a superclass. However, SystemZ uses this pattern.
class C<string self> {
  dag q = (ops !cast<C>(self));
}

def C0 : C<"C0">;

// Explore some unused corner cases.
//
// A self-reference within a class may seem icky, but it unavoidably falls out
// orthogonally of having forward class declarations and late resolve of self
// references.
class D<string self> {
  D d = !cast<D>(self);
}

def D0 : D<"D0">;

class E<E x> {
  E e = x;
}

// Putting the !cast directly in the def should work as well: we shouldn't
// depend on implementation details of when exactly the record is looked up.
//
// Note the difference between !cast<E>("E0") and plain E0: the latter wouldn't
// work here because E0 does not yet have E as a superclass while the template
// arguments are being parsed.
def E0 : E<!cast<E>("E0")>;

// Ensure that records end up with the correct type even when direct self-
// references are involved.
class Fa;
class Fb<Fa x> {
  Fa as_a = x;
}
class Fc<Fb x> {
  Fb as_b = x;
}

def F0 : Fa, Fb<F0>, Fc<F0>;
def F0x {
  Fc as_c = F0;
}
