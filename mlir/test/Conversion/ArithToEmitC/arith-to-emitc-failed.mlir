// RUN: mlir-opt -convert-arith-to-emitc %s -split-input-file -verify-diagnostics

func.func @bool(%arg0: i1, %arg1: i1) {
  // expected-error@+1 {{failed to legalize operation 'arith.addi'}}
  %0 = arith.addi %arg0, %arg1 : i1
  return
}

// -----

func.func @vector(%arg0: vector<4xi32>, %arg1: vector<4xi32>) {
  // expected-error@+1 {{failed to legalize operation 'arith.addi'}}
  %0 = arith.addi %arg0, %arg1 : vector<4xi32>
  return
}
