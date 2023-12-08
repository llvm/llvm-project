// RUN: llvm-tblgen -gen-ctags %s | FileCheck %s
// XFAIL: vg_leak

// Ensure that generated names for anonymous records are valid identifiers via the ctags index.

class foo<int X> { int THEVAL = X; }
// CHECK: {{^X }}

def : foo<2>;
// CHECK: {{^anonymous_0 }}

def X {
  foo Y = foo<1>;
}
// CHECK: {{^anonymous_1 }}
// CHECK: {{^foo }}
