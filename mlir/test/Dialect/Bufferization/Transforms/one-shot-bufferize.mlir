// RUN: mlir-opt %s -one-shot-bufferize="allow-unknown-ops" -split-input-file | FileCheck %s

// Run fuzzer with different seeds.
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-fuzzer-seed=23" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-fuzzer-seed=59" -split-input-file -o /dev/null
// RUN: mlir-opt %s -one-shot-bufferize="test-analysis-only analysis-fuzzer-seed=91" -split-input-file -o /dev/null

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
  // CHECK: memref.copy %[[A_memref]], %[[alloc]]
  // CHECK: vector.transfer_write %{{.*}}, %[[alloc]]
  // CHECK: %[[res_tensor:.*]] = bufferization.to_tensor %[[alloc]]
  %0 = vector.transfer_write %v, %A[%c0] : vector<4xf32>, tensor<?xf32>

  // CHECK: memref.dealloc %[[alloc]]
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

// -----

// CHECK-LABEL: func @read_after_write_conflict(
func @read_after_write_conflict(%cst : f32, %idx : index, %idx2 : index)
    -> (f32, f32) {
  // CHECK-DAG: %[[alloc:.*]] = memref.alloc
  // CHECK-DAG: %[[dummy:.*]] = "test.dummy_op"
  // CHECK-DAG: %[[dummy_m:.*]] = bufferization.to_memref %[[dummy]]
  %t = "test.dummy_op"() : () -> (tensor<10xf32>)

  // CHECK: memref.copy %[[dummy_m]], %[[alloc]]
  // CHECK: memref.store %{{.*}}, %[[alloc]]
  %write = tensor.insert %cst into %t[%idx2] : tensor<10xf32>

  // CHECK: %[[read:.*]] = "test.some_use"(%[[dummy]])
  %read = "test.some_use"(%t) : (tensor<10xf32>) -> (f32)
  // CHECK: %[[read2:.*]] = memref.load %[[alloc]]
  %read2 = tensor.extract %write[%idx] : tensor<10xf32>

  // CHECK: memref.dealloc %[[alloc]]
  // CHECK: return %[[read]], %[[read2]]
  return %read, %read2 : f32, f32
}

// -----

// CHECK-LABEL: func @copy_deallocated(
func @copy_deallocated() -> tensor<10xf32> {
  // CHECK: %[[alloc:.*]] = memref.alloc()
  %0 = linalg.init_tensor[10] : tensor<10xf32>
  // CHECK: %[[alloc_tensor:.*]] = bufferization.to_tensor %[[alloc]]
  // CHECK: memref.dealloc %[[alloc]]
  // CHECK: return %[[alloc_tensor]]
  return %0 : tensor<10xf32>
}

// -----

// CHECK-LABEL: func @select_different_tensors(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>
func @select_different_tensors(%t: tensor<?xf32>, %sz: index, %c: i1) -> tensor<?xf32> {
  // CHECK-DAG: %[[m:.*]] = bufferization.to_memref %[[t]] : memref<?xf32, #{{.*}}>
  // CHECK-DAG: %[[alloc:.*]] = memref.alloc(%{{.*}}) {{.*}} : memref<?xf32>
  %0 = linalg.init_tensor [%sz] : tensor<?xf32>

  // A cast must be inserted because %t and %0 have different memref types.
  // CHECK: %[[casted:.*]] = memref.cast %[[alloc]] : memref<?xf32> to memref<?xf32, #{{.*}}>
  // CHECK: arith.select %{{.*}}, %[[casted]], %[[m]]
  %1 = arith.select %c, %0, %t : tensor<?xf32>
  return %1 : tensor<?xf32>
}
