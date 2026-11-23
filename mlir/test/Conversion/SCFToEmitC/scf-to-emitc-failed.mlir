// RUN: mlir-opt -convert-scf-to-emitc %s -split-input-file -verify-diagnostics

func.func @unsupported_type_vector(%arg0 : index, %arg1 : index, %arg2 : index) -> vector<3xindex> {
  %zero = arith.constant dense<0> : vector<3xindex>
  // expected-error@+1 {{failed to legalize operation 'scf.for'}}
  %r = scf.for %i0 = %arg0 to %arg1 step %arg2 iter_args(%acc = %zero) -> vector<3xindex> {
    scf.yield %acc : vector<3xindex>
  }
  return %r : vector<3xindex>
}

// -----

// Regression test for https://github.com/llvm/llvm-project/issues/182649
// scf.while with a memref loop-carried value (which converts to emitc.array)
// must fail gracefully rather than crashing with an assertion.

func.func @while_with_memref_carried(%n: index) {
  %c0 = arith.constant 0 : index
  %zero_f64 = arith.constant 0.0 : f64
  %buf = memref.alloca() : memref<1xf64>
  memref.store %zero_f64, %buf[%c0] : memref<1xf64>
  // expected-error@+1 {{failed to legalize operation 'scf.while'}}
  scf.while (%i = %c0, %acc = %buf) : (index, memref<1xf64>) -> (index, memref<1xf64>) {
    %cond = arith.cmpi slt, %i, %n : index
    scf.condition(%cond) %i, %acc : index, memref<1xf64>
  } do {
    ^bb0(%i: index, %acc: memref<1xf64>):
    %c1 = arith.constant 1 : index
    %next_i = arith.addi %i, %c1 : index
    scf.yield %next_i, %acc : index, memref<1xf64>
  }
  return
}
