// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

class S<int s> {
  bits<2> val = !cond(!eq(s, 8):  {0, 0},
                      !eq(s, 16): 0b01,
                      !eq(s, 32): 2,
                      !eq(s, 64): {1, 1},
                              1 : ?);
}

def D8  : S<8>;
def D16 : S<16>;
def D32 : S<32>;
def D64 : S<64>;
def D128: S<128>;
// CHECK: def D128
// CHECK-NEXT: bits<2> val = { ?, ? };
// CHECK: def D16
// CHECK-NEXT: bits<2> val = { 0, 1 };
// CHECK: def D32
// CHECK-NEXT: bits<2> val = { 1, 0 };
// CHECK: def D64
// CHECK-NEXT: bits<2> val = { 1, 1 };
// CHECK: def D8
// CHECK-NEXT: bits<2> val = { 0, 0 };

