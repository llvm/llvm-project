// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: --- Defs ---

// CHECK: def X {
// CHECK:   foo Y = anonymous_0;
// CHECK: }

// CHECK: def ZD {
// CHECK:   foo Z = anonymous_1;
// CHECK: }

// CHECK: def anonymous_0 {
// CHECK:   int THEVAL = 1;
// CHECK: }

// CHECK: def anonymous_1 {
// CHECK:   int THEVAL = 42;
// CHECK: }

class foo<int X> { int THEVAL = X; }
def foo_imp : foo<1>;

def x {
  foo Y = foo_imp;    // This works.
}

def X {
  foo Y = foo<1>;     // This should work too, synthesizing a new foo<1>.
}

class Z<int X> {
  foo Z = foo<X>;
}

def ZD : Z<42>;
