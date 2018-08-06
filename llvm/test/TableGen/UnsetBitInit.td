// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

// CHECK: --- Defs ---

// Test that P and Q are not replaced by ?. TableGen's codegen emitter backend
// relies on keeping variable references like this around to describe the
// structure of instruction encodings.
//
// CHECK: def A {
// CHECK:   bits<8> Inst = { 1, 1, 1, 1, 1, 1, P, Q };
// CHECK:   bits<2> src = { ?, ? };
// CHECK:   bit P = ?;
// CHECK:   bit Q = ?;
// CHECK: }

def A {
  bits<8> Inst;
  bits<2> src;

  bit P;
  bit Q;

  let Inst{7-2} = 0x3f;
  let Inst{1} = P;
  let Inst{0} = Q;

  let P = src{1};
  let Q = src{0};
}

class x {
  field bits<32> A;
}

class y<bits<2> B> : x {
  let A{21-20} = B;
}

def z : y<{0,?}>;
