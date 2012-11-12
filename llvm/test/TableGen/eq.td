// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak
// CHECK: Value = 0
// CHECK: Value = 1

class Base<int V> {
  int Value = V;
}

class Derived<string Truth> :
  Base<!if(!eq(Truth, "true"), 1, 0)>;

def TRUE : Derived<"true">;
def FALSE : Derived<"false">;
