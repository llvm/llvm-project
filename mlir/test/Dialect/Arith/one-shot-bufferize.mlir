// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries" -split-input-file | FileCheck %s

// Run fuzzer with different seeds.
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-fuzzer-seed=23 bufferize-function-boundaries" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-fuzzer-seed=59 bufferize-function-boundaries" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-fuzzer-seed=91 bufferize-function-boundaries" -split-input-file -o /dev/null

// Test bufferization using memref types that have no layout map.
// RUN: mlir-opt %s -one-shot-bufferize="unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" -split-input-file -o /dev/null

// CHECK-LABEL: func @write_to_select_op_source
//  CHECK-SAME:     %[[t1:.*]]: memref<?xf32, strided{{.*}}>, %[[t2:.*]]: memref<?xf32, strided{{.*}}>
func.func @write_to_select_op_source(
    %t1 : tensor<?xf32> {bufferization.writable = true},
    %t2 : tensor<?xf32> {bufferization.writable = true},
    %c : i1)
  -> (tensor<?xf32>, tensor<?xf32>)
{
  %cst = arith.constant 0.0 : f32
  %idx = arith.constant 0 : index
  // CHECK: %[[alloc:.*]] = memref.alloc
  // CHECK: memref.copy %[[t1]], %[[alloc]]
  // CHECK: memref.store %{{.*}}, %[[alloc]]
  %w = tensor.insert %cst into %t1[%idx] : tensor<?xf32>
  // CHECK: %[[select:.*]] = arith.select %{{.*}}, %[[t1]], %[[t2]]
  %s = arith.select %c, %t1, %t2 : tensor<?xf32>
  // CHECK: return %[[select]], %[[alloc]]
  return %s, %w : tensor<?xf32>, tensor<?xf32>
}

// -----

// Due to the out-of-place bufferization of %t1, buffers with different layout
// maps are passed to arith.select. A cast must be inserted.

// CHECK-LABEL: func @write_after_select_read_one
//  CHECK-SAME:     %[[t1:.*]]: memref<?xf32, strided{{.*}}>, %[[t2:.*]]: memref<?xf32, strided{{.*}}>
func.func @write_after_select_read_one(
    %t1 : tensor<?xf32> {bufferization.writable = true},
    %t2 : tensor<?xf32> {bufferization.writable = true},
    %c : i1)
  -> (f32, tensor<?xf32>)
{
  %cst = arith.constant 0.0 : f32
  %idx = arith.constant 0 : index

  // CHECK: %[[alloc:.*]] = memref.alloc
  // CHECK-DAG: %[[casted:.*]] = memref.cast %[[alloc]]
  // CHECK-DAG: memref.copy %[[t1]], %[[alloc]]
  // CHECK: %[[select:.*]] = arith.select %{{.*}}, %[[casted]], %[[t2]]
  %s = arith.select %c, %t1, %t2 : tensor<?xf32>

  // CHECK: memref.store %{{.*}}, %[[select]]
  %w = tensor.insert %cst into %s[%idx] : tensor<?xf32>

  // CHECK: %[[f:.*]] = memref.load %[[t1]]
  %f = tensor.extract %t1[%idx] : tensor<?xf32>

  // CHECK: return %[[f]], %[[select]]
  return %f, %w : f32, tensor<?xf32>
}
