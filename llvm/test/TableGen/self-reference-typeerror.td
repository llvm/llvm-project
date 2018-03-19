// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s
// XFAIL: vg_leak

class A<A x> {
  A a = x;
}

// At the time A0 is referenced, A has not yet been established as a superclass.
// This kind of self-reference is discourage, but if you *really* want it, you
// can force it with !cast.
//
// CHECK: Value 'A:x' of type 'A' is incompatible with initializer
def A0 : A<A0>;
