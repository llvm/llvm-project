// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: --- Defs ---

// CHECK: def A0 {
// CHECK:   bit eq = 1;
// CHECK:   bit ne = 0;
// CHECK:   bit le = 1;
// CHECK:   bit lt = 0;
// CHECK:   bit ge = 1;
// CHECK:   bit gt = 0;
// CHECK: }

// CHECK: def A1 {
// CHECK:   bit eq = 0;
// CHECK:   bit ne = 1;
// CHECK:   bit le = 1;
// CHECK:   bit lt = 1;
// CHECK:   bit ge = 0;
// CHECK:   bit gt = 0;
// CHECK: }

// CHECK: def A2 {
// CHECK:   bit eq = 0;
// CHECK:   bit ne = 1;
// CHECK:   bit le = 0;
// CHECK:   bit lt = 0;
// CHECK:   bit ge = 1;
// CHECK:   bit gt = 1;
// CHECK: }

// CHECK: def A3 {
// CHECK:   bit eq = 0;
// CHECK:   bit ne = 1;
// CHECK:   bit le = 0;
// CHECK:   bit lt = 0;
// CHECK:   bit ge = 1;
// CHECK:   bit gt = 1;
// CHECK: }

class A<int x, int y> {
  bit eq = !eq(x, y);
  bit ne = !ne(x, y);
  bit le = !le(x, y);
  bit lt = !lt(x, y);
  bit ge = !ge(x, y);
  bit gt = !gt(x, y);
}

def A0 : A<-3, -3>;
def A1 : A<-1, 4>;
def A2 : A<3, -2>;
def A3 : A<4, 2>;
