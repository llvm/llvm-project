
// RUN: not llvm-tblgen %s 2>&1 | FileCheck %s -DFILE=%s

def a {
  bits<2> opc = { 0, 1 };
  bits<2> opc2 = { 1, 0 };
  bits<1> opc3 = { 1 };
  // CHECK: [[FILE]]:[[@LINE+1]]:15: error: Field 'a' of type 'bits<2>' is incompatible with value '{ opc{1}, opc{0}, opc2{1}, opc2{0} }' of type bit initializer with length 4
  bits<2> a = { opc, opc2 }; // error!
}

def {
  // CHECK: [[FILE]]:[[@LINE+1]]:16: error: Field 'B1' of type 'bits<2>' is incompatible with value '{ 0, 1, 1 }' of type bit initializer with length 3
  bits<2> B1 = 0b011;  // bitfield is too small, reject

  // CHECK: [[FILE]]:[[@LINE+1]]:16: error: Field 'C1' of type 'bits<2>' is incompatible with value '{ 1, 1, 1 }' of type bit initializer with length 3
  bits<2> C1 = 0b111;  // bitfield is too small, reject

  // CHECK: [[FILE]]:[[@LINE+1]]:16: error: Field 'D3' of type 'bits<3>' is incompatible with value '{ 0, 0 }' of type bit initializer with length 2
  bits<3> D3 = { 0, 0 }; // type mismatch.  RHS doesn't have enough bits

  // CHECK: [[FILE]]:[[@LINE+1]]:16: error: Field 'D4' of type 'bits<3>' is incompatible with value '{ 0, 0 }' of type bit initializer with length 2
  bits<3> D4 = { 0b00 }; // type mismatch.  RHS doesn't have enough bits

  bits<1> D7 = { 3 };    // type mismatch.  LHS doesn't have enough bits

  // CHECK: [[FILE]]:[[@LINE+1]]:16: error: Field 'D8' of type 'bits<2>' is incompatible with value '{ 0 }' of type bit initializer with length 1
  bits<2> D8 = { 0 };    // type mismatch.  RHS doesn't have enough bits

  // CHECK: [[FILE]]:[[@LINE+1]]:16: error: Field 'F2' of type 'bits<7>' is incompatible with value '{ 0, 1, 1, 0, 0, 1, 0, 0 }' of type bit initializer with length 8
  bits<7> F2 = { 0, 1, 0b1001, 0, 0b0 }; // LHS doesn't have enough bits

  // CHECK: [[FILE]]:[[@LINE+1]]:16: error: Field 'F3' of type 'bits<9>' is incompatible with value '{ 0, 1, 1, 0, 0, 1, 0, 0 }' of type bit initializer with length 8
  bits<9> F3 = { 0, 1, 0b1001, 0, 0b0 }; // RHS doesn't have enough bits

  // CHECK: Initializer of 'D7' in 'anonymous_0' could not be fully resolved: { !cast<bit>(3) }
}
