// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: --- Defs ---

// CHECK: def A0 {
// CHECK:   bits<8> add = { 0, 0, 0, 1, 1, 0, 0, 0 };
// CHECK:   bits<8> sub = { 0, 0, 0, 1, 0, 0, 1, 0 };
// CHECK:   bits<8> and = { 0, 0, 0, 0, 0, 0, 0, 1 };
// CHECK:   bits<8> or = { 0, 0, 0, 1, 0, 1, 1, 1 };
// CHECK:   bits<8> xor = { 0, 0, 0, 1, 0, 1, 1, 0 };
// CHECK:   bits<8> srl = { 0, 0, 0, 0, 0, 0, 1, 0 };
// CHECK:   bits<8> sra = { 0, 0, 0, 0, 0, 0, 1, 0 };
// CHECK:   bits<8> shl = { 1, 0, 1, 0, 1, 0, 0, 0 };

// CHECK:   bits<8> sra = { 1, 1, 1, 1, 1, 1, 1, 1 };

class A<bits<8> a, bits<2> b> {
  // Operands of different bits types are allowed.
  bits<8> add = !add(a, b);
  bits<8> sub = !sub(a, b);
  bits<8> and = !and(a, b);
  bits<8> or =  !or(a, b);
  bits<8> xor = !xor(a, b);
  bits<8> srl = !srl(a, b);
  bits<8> sra = !sra(a, b);
  bits<8> shl = !shl(a, b);
}

def A0 : A<21, 3>;

def A1 {
  bits<8> sra = !sra(-1, 3);
}

