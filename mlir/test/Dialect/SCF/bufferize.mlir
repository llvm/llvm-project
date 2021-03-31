// RUN: mlir-opt %s -scf-bufferize | FileCheck %s

// CHECK-LABEL:   func @if(
// CHECK-SAME:             %[[PRED:.*]]: i1,
// CHECK-SAME:             %[[TRUE_TENSOR:.*]]: tensor<?xf32>,
// CHECK-SAME:             %[[FALSE_TENSOR:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           %[[RESULT_MEMREF:.*]] = scf.if %[[PRED]] -> (memref<?xf32>) {
// CHECK:             %[[TRUE_MEMREF:.*]] = memref.buffer_cast %[[TRUE_TENSOR]] : memref<?xf32>
// CHECK:             scf.yield %[[TRUE_MEMREF]] : memref<?xf32>
// CHECK:           } else {
// CHECK:             %[[FALSE_MEMREF:.*]] = memref.buffer_cast %[[FALSE_TENSOR]] : memref<?xf32>
// CHECK:             scf.yield %[[FALSE_MEMREF]] : memref<?xf32>
// CHECK:           }
// CHECK:           %[[RESULT_TENSOR:.*]] = memref.tensor_load %[[RESULT_MEMREF:.*]] : memref<?xf32>
// CHECK:           return %[[RESULT_TENSOR]] : tensor<?xf32>
// CHECK:         }
func @if(%pred: i1, %true_val: tensor<?xf32>, %false_val: tensor<?xf32>) -> tensor<?xf32> {
  %0 = scf.if %pred -> (tensor<?xf32>) {
    scf.yield %true_val : tensor<?xf32>
  } else {
    scf.yield %false_val : tensor<?xf32>
  }
  return %0 : tensor<?xf32>
}

// CHECK-LABEL:   func @for(
// CHECK-SAME:              %[[TENSOR:.*]]: tensor<f32>,
// CHECK-SAME:              %[[LB:.*]]: index, %[[UB:.*]]: index,
// CHECK-SAME:              %[[STEP:.*]]: index) -> tensor<f32> {
// CHECK:           %[[MEMREF:.*]] = memref.buffer_cast %[[TENSOR]] : memref<f32>
// CHECK:           %[[RESULT_MEMREF:.*]] = scf.for %[[VAL_6:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[ITER:.*]] = %[[MEMREF]]) -> (memref<f32>) {
// CHECK:             %[[TENSOR_ITER:.*]] = memref.tensor_load %[[ITER]] : memref<f32>
// CHECK:             %[[MEMREF_YIELDED:.*]] = memref.buffer_cast %[[TENSOR_ITER]] : memref<f32>
// CHECK:             scf.yield %[[MEMREF_YIELDED]] : memref<f32>
// CHECK:           }
// CHECK:           %[[VAL_8:.*]] = memref.tensor_load %[[VAL_9:.*]] : memref<f32>
// CHECK:           return %[[VAL_8]] : tensor<f32>
// CHECK:         }
func @for(%arg0: tensor<f32>, %lb: index, %ub: index, %step: index) -> tensor<f32> {
  %ret = scf.for %iv = %lb to %ub step %step iter_args(%iter = %arg0) -> tensor<f32> {
    scf.yield %iter : tensor<f32>
  }
  return %ret : tensor<f32>
}

// Check whether this converts at all.
//
// It would previously fail altogether.
// CHECK-LABEL:   func @if_correct_recursive_legalization_behavior
// CHECK: "test.munge_tensor"
func @if_correct_recursive_legalization_behavior(%pred: i1, %tensor: tensor<f32>) -> tensor<f32> {
  %0 = scf.if %pred -> (tensor<f32>) {
    %1 = "test.munge_tensor"(%tensor) : (tensor<f32>) -> (tensor<f32>)
    scf.yield %1: tensor<f32>
  } else {
    %1 = "test.munge_tensor"(%tensor) : (tensor<f32>) -> (tensor<f32>)
    scf.yield %1 : tensor<f32>
  }
  return %0 : tensor<f32>
}

// CHECK-LABEL:   func @for_correct_recursive_legalization_behavior(
// CHECK-SAME:                                                      %[[TENSOR:.*]]: tensor<f32>,
// CHECK-SAME:                                                      %[[INDEX:.*]]: index) -> tensor<f32> {
// CHECK:           %[[MEMREF:.*]] = memref.buffer_cast %[[TENSOR]] : memref<f32>
// CHECK:           %[[RESULT:.*]] = scf.for %[[IV:.*]] = %[[INDEX]] to %[[INDEX]] step %[[INDEX]] iter_args(%[[MEMREF_ITER:.*]] = %[[MEMREF]]) -> (memref<f32>) {
// CHECK:             %[[TENSOR_ITER:.*]] = memref.tensor_load %[[MEMREF_ITER]] : memref<f32>
// CHECK:             %[[TENSOR_MUNGED:.*]] = "test.munge_tensor"(%[[TENSOR_ITER]]) : (tensor<f32>) -> tensor<f32>
// CHECK:             %[[MEMREF_MUNGED:.*]] = memref.buffer_cast %[[TENSOR_MUNGED]] : memref<f32>
// CHECK:             scf.yield %[[MEMREF_MUNGED]] : memref<f32>
// CHECK:           }
// CHECK:           %[[TENSOR:.*]] = memref.tensor_load %[[RESULT:.*]] : memref<f32>
// CHECK:           return %[[TENSOR]] : tensor<f32>
// CHECK:         }
func @for_correct_recursive_legalization_behavior(%arg0: tensor<f32>, %index: index) -> tensor<f32> {
  %ret = scf.for %iv = %index to %index step %index iter_args(%iter = %arg0) -> tensor<f32> {
    %0 = "test.munge_tensor"(%iter) : (tensor<f32>) -> (tensor<f32>)
    scf.yield %0 : tensor<f32>
  }
  return %ret : tensor<f32>
}
