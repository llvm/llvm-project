// RUN: mlir-opt %s -scf-bufferize | FileCheck %s

// CHECK-LABEL:   func @if(
// CHECK-SAME:             %[[PRED:.*]]: i1,
// CHECK-SAME:             %[[TRUE_TENSOR:.*]]: tensor<?xf32>,
// CHECK-SAME:             %[[FALSE_TENSOR:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           %[[RESULT_MEMREF:.*]] = scf.if %[[PRED]] -> (memref<?xf32>) {
// CHECK:             %[[TRUE_MEMREF:.*]] = tensor_to_memref %[[TRUE_TENSOR]] : memref<?xf32>
// CHECK:             scf.yield %[[TRUE_MEMREF]] : memref<?xf32>
// CHECK:           } else {
// CHECK:             %[[FALSE_MEMREF:.*]] = tensor_to_memref %[[FALSE_TENSOR]] : memref<?xf32>
// CHECK:             scf.yield %[[FALSE_MEMREF]] : memref<?xf32>
// CHECK:           }
// CHECK:           %[[RESULT_TENSOR:.*]] = tensor_load %[[RESULT_MEMREF:.*]] : memref<?xf32>
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
// CHECK:           %[[MEMREF:.*]] = tensor_to_memref %[[TENSOR]] : memref<f32>
// CHECK:           %[[RESULT_MEMREF:.*]] = scf.for %[[VAL_6:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[ITER:.*]] = %[[MEMREF]]) -> (memref<f32>) {
// CHECK:             scf.yield %[[ITER]] : memref<f32>
// CHECK:           }
// CHECK:           %[[VAL_8:.*]] = tensor_load %[[VAL_9:.*]] : memref<f32>
// CHECK:           return %[[VAL_8]] : tensor<f32>
// CHECK:         }
func @for(%arg0: tensor<f32>, %lb: index, %ub: index, %step: index) -> tensor<f32> {
  %ret = scf.for %iv = %lb to %ub step %step iter_args(%iter = %arg0) -> tensor<f32> {
    scf.yield %iter : tensor<f32>
  }
  return %ret : tensor<f32>
}
