// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: --- Defs ---

// CHECK: def A1 {
// CHECK:   int Val = 0;
// CHECK: }

// CHECK: def A2 {
// CHECK:   int Val = 3;
// CHECK: }

// CHECK: def B1 {
// CHECK:   int Val = 0;
// CHECK: }

// CHECK: def B2 {
// CHECK:   int Val = 2;
// CHECK: }

class A<list<int> L> {
  int Val = !size(L);
}

class B<list<string> L> {
  int Val = !size(L);
}

def A1 : A<[]>;
def A2 : A<[1, 1, 2]>;

def B1 : B<[]>;
def B2 : B<["a", "b"]>;
