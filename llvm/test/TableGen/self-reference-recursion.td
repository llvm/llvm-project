// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s
// XFAIL: vg_leak

class A<string self> {
  int x = !cast<A>(self).x;
}

// CHECK: error: Attempting to access field 'x' of 'A0' is a forbidden self-reference
def A0 : A<"A0">;
