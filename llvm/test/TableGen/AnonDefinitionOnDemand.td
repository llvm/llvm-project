// RUN: llvm-tblgen < %s
// XFAIL: vg_leak

class foo<int X> { int THEVAL = X; }
def foo_imp : foo<1>;

def x {
  foo Y = foo_imp;    // This works.
}

def X {
  foo Y = foo<1>;     // This should work too, synthesizing a new foo<1>.
}
