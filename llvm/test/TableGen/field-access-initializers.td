// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: class B<A B:impl = ?> {
// CHECK:   string value = B:impl.value;
// CHECK: }

// CHECK: --- Defs ---

// CHECK: def A1 {
// CHECK:   string value = ?;
// CHECK: }

// CHECK: def B1 {
// CHECK:   string value = ?;
// CHECK: }

class A {
  string value = ?;
}

class B<A impl> : A {
  let value = impl.value;
}

def A1 : A;
def B1 : B<A1>;
