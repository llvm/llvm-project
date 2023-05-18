// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK-LABEL: def FALSE {
//       CHECK:   int Value = 0;
//       CHECK: }

// CHECK-LABEL: def TRUE {
//       CHECK:   int Value = 1;
//       CHECK: }

// CHECK-LABEL: def X_NE {
//       CHECK:   bit a = 1;
//       CHECK:   bit b = 0;
//       CHECK: }

class Base<int V> {
  int Value = V;
}

class Derived<string Truth> :
  Base<!if(!eq(Truth, "true"), 1, 0)>;

def TRUE : Derived<"true">;
def FALSE : Derived<"false">;

def X_NE {
  bit a = !ne("true", "false");
  bit b = !ne("foo", "foo");
}
