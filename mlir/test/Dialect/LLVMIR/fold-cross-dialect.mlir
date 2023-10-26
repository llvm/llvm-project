// RUN: mlir-opt %s --test-constant-fold --split-input-file | FileCheck %s

// CHECK-LABEL: llvm.func @fold_constant_mismatching_type
llvm.func @fold_constant_mismatching_type() -> i64 {
  // CHECK-DAG: llvm.mlir.constant(42
  // CHECK-DAG: llvm.mlir.constant(2
  %0 = llvm.mlir.constant(42 : index) : i64
  %1 = llvm.mlir.constant(2 : i64) : i64
  // Using arith.add as there is no folder for llvm.add.
  // This is not expected to fold because attribute types differ.
  // CHECK: arith.add
  %2 = arith.addi %0, %1 : i64
  llvm.return %2 : i64
}

// -----

// CHECK-LABEL: @fold_constant_matching_type
llvm.func @fold_constant_matching_type() -> i64 {
  // CHECK-NOT: llvm.mlir.constant
  %0 = llvm.mlir.constant(42 : i64) : i64
  %1 = llvm.mlir.constant(2 : i64) : i64
  // Using arith.add as there is no folder for llvm.add.
  // This is expected to fold.
  // CHECK: %[[V:.+]] = arith.constant 44 : i64
  // CHECK: llvm.return %[[V]]
  %2 = arith.addi %0, %1 : i64
  llvm.return %2 : i64
}

// -----

// CHECK-LABEL: @fold_constant_vector_mismatching_type
func.func @fold_constant_vector_mismatching_type() {
  // CHECK: llvm.mlir.constant(0
  %220 = llvm.mlir.constant(0 : index) : i64
  // CHECK: vector.broadcast
  %365 = vector.broadcast %220 : i64 to vector<26xi64>
  return
}

// -----

// CHECK-LABEL: @fold_constant_vector_matching_type
func.func @fold_constant_vector_matching_type() -> vector<26xi64>{
  // CHECK: arith.constant dense<0> : vector<26xi64>
  %220 = llvm.mlir.constant(0 : i64) : i64
  %365 = vector.broadcast %220 : i64 to vector<26xi64>
  return %365 : vector<26xi64>
}
