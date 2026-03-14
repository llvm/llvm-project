// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s
// XFAIL: vg_leak

class Cl<Cl rec> {
  Cl Arec = rec;
}

// At the time A0 is referenced, A has not yet been established as a superclass.
// This kind of self-reference is discourage, but if you *really* want it, you
// can force it with !cast.
//
// CHECK: alue specified for template argument 'Cl:rec'

def Rec0 : Cl<Rec0>;
