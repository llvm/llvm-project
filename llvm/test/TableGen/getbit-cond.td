// RUN: llvm-tblgen %s | FileCheck %s

class C<bit val> {
  bits<4> word;
  bit flag = 1;
  let word{0} = !cond(val : flag);
}

class D<bit val> : C<val>;

def One : D<1>;

// CHECK: class D
// CHECK-NEXT: bits<4> word = { ?, ?, ?, !cond(D:val: flag) };
// CHECK-NEXT: bit flag = 1;

// CHECK: def One
// CHECK-NEXT: bits<4> word = { ?, ?, ?, 1 };
// CHECK-NEXT: bit flag = 1;
