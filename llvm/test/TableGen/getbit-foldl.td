// RUN: llvm-tblgen %s | FileCheck %s

class C<bit val, list<bit> lst> {
  bits<4> word;
  bit flag = 1;
  let word{0} = !foldl(val, lst, a, b, !cast<bit>(!or(a, b)));
}

class D<bit val, list<bit> lst> : C<val, lst>;

def One : D<0, [1, 0, 1]>;

// CHECK: class D
// CHECK-NEXT: bits<4> word = { ?, ?, ?, !foldl(D:val, D:lst, a, b, !cast<bit>(!or(a, b))) };
// CHECK-NEXT: bit flag = 1;

// CHECK: def One
// CHECK-NEXT: bits<4> word = { ?, ?, ?, 1 };
// CHECK-NEXT: bit flag = 1;
