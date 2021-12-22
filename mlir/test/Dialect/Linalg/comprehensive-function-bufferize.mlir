// RUN: mlir-opt %s -test-comprehensive-function-bufferize="allow-return-memref allow-unknown-ops" -split-input-file | FileCheck %s

// Run fuzzer with different seeds.
// RUN: mlir-opt %s -test-comprehensive-function-bufferize="test-analysis-only analysis-fuzzer-seed=23" -split-input-file -o /dev/null
// RUN: mlir-opt %s -test-comprehensive-function-bufferize="test-analysis-only analysis-fuzzer-seed=59" -split-input-file -o /dev/null
// RUN: mlir-opt %s -test-comprehensive-function-bufferize="test-analysis-only analysis-fuzzer-seed=91" -split-input-file -o /dev/null

// CHECK-LABEL: func @use_tensor_func_arg(
//  CHECK-SAME:     %[[A:.*]]: tensor<?xf32>
func @use_tensor_func_arg(%A : tensor<?xf32>) -> (vector<4xf32>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32

  // CHECK: %[[A_memref:.*]] = bufferization.to_memref %[[A]]
  // CHECK: %[[res:.*]] = vector.transfer_read %[[A_memref]]
  %0 = vector.transfer_read %A[%c0], %f0 : tensor<?xf32>, vector<4xf32>

  // CHECK: return %[[res]]
  return %0 : vector<4xf32>
}

// -----

// CHECK-LABEL: func @return_tensor(
//  CHECK-SAME:     %[[A:.*]]: tensor<?xf32>
func @return_tensor(%A : tensor<?xf32>, %v : vector<4xf32>) -> (tensor<?xf32>) {
  %c0 = arith.constant 0 : index

  // CHECK: %[[A_memref:.*]] = bufferization.to_memref %[[A]]
  // CHECK: %[[dim:.*]] = tensor.dim %[[A]]
  // CHECK: %[[alloc:.*]] = memref.alloc(%[[dim]])
  // CHECK: %[[casted:.*]] = memref.cast %[[alloc]]
  // CHECK: %[[res_tensor:.*]] = bufferization.to_tensor %[[casted]]
  // CHECK: memref.copy %[[A_memref]], %[[casted]]
  // CHECK: vector.transfer_write %{{.*}}, %[[alloc]]
  %0 = vector.transfer_write %v, %A[%c0] : vector<4xf32>, tensor<?xf32>

  // CHECK: return %[[res_tensor]]
  return %0 : tensor<?xf32>
}

// -----

// CHECK-LABEL: func @func_without_tensor_args
func @func_without_tensor_args(%v : vector<10xf32>) -> () {
  // CHECK: %[[alloc:.*]] = memref.alloc()
  %0 = linalg.init_tensor[10] : tensor<10xf32>

  %c0 = arith.constant 0 : index
  // CHECK: vector.transfer_write %{{.*}}, %[[alloc]]
  %1 = vector.transfer_write %v, %0[%c0] : vector<10xf32>, tensor<10xf32>

  %cst = arith.constant 0.0 : f32
  // CHECK: vector.transfer_read %[[alloc]]
  %r = vector.transfer_read %1[%c0], %cst : tensor<10xf32>, vector<11xf32>

  vector.print %r : vector<11xf32>
  return
}

// -----

// CHECK-LABEL: func private @private_func
func private @private_func(tensor<?xf32>) -> ()

// CHECK-LABEL: func @empty_func()
func @empty_func() -> () {
  return
}
