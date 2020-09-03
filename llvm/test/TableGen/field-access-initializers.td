// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: --- Defs ---

// CHECK: def A1 {
// CHECK:   string value = ?;
// CHECK: }

// CHECK: def B1 {
// CHECK:   string value = A1.value;
// CHECK: }

class A {
  string value = ?;
}

class B<A impl> : A {
  let value = impl.value;
}

def A1 : A;
def B1 : B<A1>;
