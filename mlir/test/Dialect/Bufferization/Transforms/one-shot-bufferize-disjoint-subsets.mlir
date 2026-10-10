// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries" -split-input-file | FileCheck %s
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only bufferize-function-boundaries" -split-input-file | FileCheck %s --check-prefix=ANALYSIS
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-heuristic=fuzzer analysis-fuzzer-seed=23 bufferize-function-boundaries" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-heuristic=fuzzer analysis-fuzzer-seed=59 bufferize-function-boundaries" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-heuristic=fuzzer analysis-fuzzer-seed=91 bufferize-function-boundaries" -split-input-file -o /dev/null

// The inner extraction cannot be compared directly with the insertion. The
// outer extraction still bounds the read and is disjoint from the insertion.

// CHECK-LABEL: func @nested_read(
// CHECK-NOT: memref.alloc
// CHECK: memref.copy
// CHECK: memref.subview
// CHECK: memref.subview
// CHECK-NOT: memref.alloc
// CHECK: return
// ANALYSIS-LABEL: func @nested_read(
// ANALYSIS: tensor.insert_slice
// ANALYSIS-SAME: __inplace_operands_attr__ = ["true", "true"]
func.func @nested_read(
    %t: tensor<8xf32> {bufferization.writable = true}, %s: tensor<4xf32>)
    -> (tensor<8xf32>, tensor<2xf32>) {
  %written = tensor.insert_slice %s into %t[0][4][1]
      : tensor<4xf32> into tensor<8xf32>
  %outer = tensor.extract_slice %t[4][4][1]
      : tensor<8xf32> to tensor<4xf32>
  %inner = tensor.extract_slice %outer[0][2][1]
      : tensor<4xf32> to tensor<2xf32>
  return %written, %inner : tensor<8xf32>, tensor<2xf32>
}

// -----

// The same containment reasoning applies on the write side.

// CHECK-LABEL: func @nested_write(
// CHECK-NOT: memref.alloc
// CHECK: linalg.fill
// CHECK-NOT: memref.alloc
// CHECK: return
// ANALYSIS-LABEL: func @nested_write(
// ANALYSIS: linalg.fill
// ANALYSIS-SAME: __inplace_operands_attr__ = ["none", "true"]
func.func @nested_write(
    %t: tensor<8xf32> {bufferization.writable = true}, %value: f32)
    -> (tensor<2xf32>, tensor<4xf32>) {
  %outer = tensor.extract_slice %t[0][4][1]
      : tensor<8xf32> to tensor<4xf32>
  %inner = tensor.extract_slice %outer[0][2][1]
      : tensor<4xf32> to tensor<2xf32>
  %rhs = tensor.extract_slice %t[4][4][1]
      : tensor<8xf32> to tensor<4xf32>
  %written = linalg.fill ins(%value : f32) outs(%inner : tensor<2xf32>)
      -> tensor<2xf32>
  return %written, %rhs : tensor<2xf32>, tensor<4xf32>
}

// -----

// Neither the inner nor outer bound proves disjointness in this case.

// CHECK-LABEL: func @nested_overlap(
// CHECK: memref.alloc
// CHECK: linalg.fill
// ANALYSIS-LABEL: func @nested_overlap(
// ANALYSIS: tensor.extract_slice
// ANALYSIS-SAME: __inplace_operands_attr__ = ["false"]
func.func @nested_overlap(
    %t: tensor<8xf32> {bufferization.writable = true}, %value: f32)
    -> (tensor<2xf32>, tensor<4xf32>) {
  %outer = tensor.extract_slice %t[0][4][1]
      : tensor<8xf32> to tensor<4xf32>
  %inner = tensor.extract_slice %outer[0][2][1]
      : tensor<4xf32> to tensor<2xf32>
  %rhs = tensor.extract_slice %t[1][4][1]
      : tensor<8xf32> to tensor<4xf32>
  %written = linalg.fill ins(%value : f32) outs(%inner : tensor<2xf32>)
      -> tensor<2xf32>
  return %written, %rhs : tensor<2xf32>, tensor<4xf32>
}

// -----

// In iteration i the write to t[i] and read from t[i-1] are disjoint, but the
// read must not observe the write from iteration i-1. The fill needs a private
// buffer; tensor SSA semantics keep the original t unchanged across iterations.

// CHECK-LABEL: func @cross_iteration_fill(
// CHECK-SAME: %[[T:.*]]: memref<4xf32,
// CHECK: scf.for
// CHECK: %[[ALLOC:.*]] = memref.alloc
// CHECK: %[[RHS:.*]] = memref.subview %[[T]]
// CHECK: linalg.fill {{.*}} outs(%[[ALLOC]]
// CHECK: memref.load %[[RHS]]
// ANALYSIS-LABEL: func @cross_iteration_fill(
// ANALYSIS: tensor.extract_slice
// ANALYSIS-SAME: __inplace_operands_attr__ = ["false", "none"]
func.func @cross_iteration_fill(
    %t: tensor<4xf32> {bufferization.writable = true}) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  %value = arith.constant 100.0 : f32
  %result = scf.for %i = %c1 to %c4 step %c1
      iter_args(%sum = %zero) -> f32 {
    %prev = arith.subi %i, %c1 : index
    %lhs = tensor.extract_slice %t[%i][1][1]
        : tensor<4xf32> to tensor<1xf32>
    %rhs = tensor.extract_slice %t[%prev][1][1]
        : tensor<4xf32> to tensor<1xf32>
    %written = linalg.fill ins(%value : f32) outs(%lhs : tensor<1xf32>)
        -> tensor<1xf32>
    %new = tensor.extract %written[%c0] : tensor<1xf32>
    %old = tensor.extract %rhs[%c0] : tensor<1xf32>
    %pair = arith.addf %new, %old : f32
    %next = arith.addf %sum, %pair : f32
    scf.yield %next : f32
  }
  return %result : f32
}

// -----

// Exact insertion subsets must obey the same cross-iteration restriction.

// CHECK-LABEL: func @cross_iteration_insert(
// CHECK-SAME: %[[T:.*]]: memref<4xf32,
// CHECK: scf.for
// CHECK: %[[ALLOC:.*]] = memref.alloc
// CHECK: memref.copy %[[T]], %[[ALLOC]]
// ANALYSIS-LABEL: func @cross_iteration_insert(
// ANALYSIS: tensor.insert_slice
// ANALYSIS-SAME: __inplace_operands_attr__ = ["true", "false", "none"]
func.func @cross_iteration_insert(
    %t: tensor<4xf32> {bufferization.writable = true}, %s: tensor<1xf32>)
    -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  %result = scf.for %i = %c1 to %c4 step %c1
      iter_args(%sum = %zero) -> f32 {
    %prev = arith.subi %i, %c1 : index
    %written = tensor.insert_slice %s into %t[%i][1][1]
        : tensor<1xf32> into tensor<4xf32>
    %rhs = tensor.extract_slice %t[%prev][1][1]
        : tensor<4xf32> to tensor<1xf32>
    %new = tensor.extract %written[%i] : tensor<4xf32>
    %old = tensor.extract %rhs[%c0] : tensor<1xf32>
    %pair = arith.addf %new, %old : f32
    %next = arith.addf %sum, %pair : f32
    scf.yield %next : f32
  }
  return %result : f32
}

// -----

// Unstructured loops have the same cross-iteration hazard.

// CHECK-LABEL: func @cross_iteration_cfg(
// CHECK-SAME: %[[T:.*]]: memref<4xf32,
// CHECK: %[[ALLOC:.*]] = memref.alloc
// CHECK: %[[RHS:.*]] = memref.subview %[[T]]
// CHECK: linalg.fill {{.*}} outs(%[[ALLOC]]
// CHECK: memref.load %[[RHS]]
// ANALYSIS-LABEL: func @cross_iteration_cfg(
// ANALYSIS: tensor.extract_slice
// ANALYSIS-SAME: __inplace_operands_attr__ = ["false", "none"]
func.func @cross_iteration_cfg(
    %t: tensor<4xf32> {bufferization.writable = true}) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  %value = arith.constant 100.0 : f32
  cf.br ^loop(%c1, %zero : index, f32)
^loop(%i: index, %sum: f32):
  %prev = arith.subi %i, %c1 : index
  %lhs = tensor.extract_slice %t[%i][1][1]
      : tensor<4xf32> to tensor<1xf32>
  %rhs = tensor.extract_slice %t[%prev][1][1]
      : tensor<4xf32> to tensor<1xf32>
  %written = linalg.fill ins(%value : f32) outs(%lhs : tensor<1xf32>)
      -> tensor<1xf32>
  %new = tensor.extract %written[%c0] : tensor<1xf32>
  %old = tensor.extract %rhs[%c0] : tensor<1xf32>
  %pair = arith.addf %new, %old : f32
  %next = arith.addf %sum, %pair : f32
  %next_i = arith.addi %i, %c1 : index
  %continue = arith.cmpi ult, %next_i, %c4 : index
  cf.cond_br %continue, ^loop(%next_i, %next : index, f32), ^exit(%next : f32)
^exit(%result: f32):
  return %result : f32
}

// -----

// A fresh definition in each iteration resets the read/write sequence. Merely
// being inside a loop must not disable the disjoint-subset optimization.

// CHECK-LABEL: func @fresh_definition_in_loop(
// CHECK: scf.for
// CHECK: memref.alloc
// CHECK-NOT: memref.alloc
// CHECK: linalg.fill
// CHECK-NOT: memref.alloc
// CHECK: scf.yield
// ANALYSIS-LABEL: func @fresh_definition_in_loop(
// ANALYSIS: tensor.extract_slice
// ANALYSIS-SAME: __inplace_operands_attr__ = ["true"]
// ANALYSIS: tensor.extract_slice
// ANALYSIS-SAME: __inplace_operands_attr__ = ["true"]
// ANALYSIS: linalg.fill
// ANALYSIS-SAME: __inplace_operands_attr__ = ["none", "true"]
func.func @fresh_definition_in_loop(%value: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %zero = arith.constant 0.0 : f32
  %result = scf.for %i = %c1 to %c4 step %c1
      iter_args(%sum = %zero) -> f32 {
    %t = tensor.from_elements %value, %value, %value, %value : tensor<4xf32>
    %lhs = tensor.extract_slice %t[0][2][1]
        : tensor<4xf32> to tensor<2xf32>
    %rhs = tensor.extract_slice %t[2][2][1]
        : tensor<4xf32> to tensor<2xf32>
    %written = linalg.fill ins(%zero : f32) outs(%lhs : tensor<2xf32>)
        -> tensor<2xf32>
    %new = tensor.extract %written[%c0] : tensor<2xf32>
    %old = tensor.extract %rhs[%c0] : tensor<2xf32>
    %pair = arith.addf %new, %old : f32
    %next = arith.addf %sum, %pair : f32
    scf.yield %next : f32
  }
  return %result : f32
}
