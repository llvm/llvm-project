// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

class C<list<bits<4>> L> {
  bits<2> V0 = L[0]{1...0};
  bits<2> V1 = L[1]{3...2};
  string V2 = !if(L[0]{0}, "Odd", "Even");
}

def c0 : C<[0b0101, 0b1010]>;

// CHECK: def c0
// CHECK-NEXT: bits<2> V0 = { 0, 1 };
// CHECK-NEXT: bits<2> V1 = { 1, 0 };
// CHECK-NEXT: string V2 = "Odd";
