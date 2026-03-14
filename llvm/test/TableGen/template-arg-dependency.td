// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: --- Defs ---

// CHECK: def A0 {
// CHECK:   int ret = 3;
// CHECK: }

class A<int a,
        int b = !add(a, 1),
        int sum = !add(a, b)> {
  int ret = sum;
}

def A0 : A<1>;
