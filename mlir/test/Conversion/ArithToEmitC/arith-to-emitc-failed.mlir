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

// -----

func.func @unsuppoted_emitc_type(%arg0: i4, %arg1: i4) {
  // expected-error@+1 {{failed to legalize operation 'arith.addi'}}
  %0 = arith.addi %arg0, %arg1 : i4
  return
}

// -----

func.func private @rank0_constant() -> memref<i64> {
  // expected-error@+1 {{failed to legalize operation 'arith.constant'}}
  %0 = arith.constant dense<-1> : memref<i64>
  return %0 : memref<i64>
}

// -----

func.func private @rank1_constant() -> memref<1xi64> {
  // expected-error@+1 {{failed to legalize operation 'arith.constant'}}
  %0 = arith.constant dense<[-1]> : memref<1xi64>
  return %0 : memref<1xi64>
}
