// RUN: mlir-opt %s -scf-bufferize | FileCheck %s

// CHECK-LABEL:   func @if(
// CHECK-SAME:             %[[PRED:.*]]: i1,
// CHECK-SAME:             %[[TRUE_TENSOR:.*]]: tensor<?xf32>,
// CHECK-SAME:             %[[FALSE_TENSOR:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           %[[TRUE_MEMREF:.*]] = bufferization.to_memref %[[TRUE_TENSOR]] : memref<?xf32>
// CHECK:           %[[FALSE_MEMREF:.*]] = bufferization.to_memref %[[FALSE_TENSOR]] : memref<?xf32>
// CHECK:           %[[RESULT_MEMREF:.*]] = scf.if %[[PRED]] -> (memref<?xf32>) {
// CHECK:             scf.yield %[[TRUE_MEMREF]] : memref<?xf32>
// CHECK:           } else {
// CHECK:             scf.yield %[[FALSE_MEMREF]] : memref<?xf32>
// CHECK:           }
// CHECK:           %[[RESULT_TENSOR:.*]] = bufferization.to_tensor %[[RESULT_MEMREF:.*]] : memref<?xf32>
// CHECK:           return %[[RESULT_TENSOR]] : tensor<?xf32>
// CHECK:         }
func.func @if(%pred: i1, %true_val: tensor<?xf32>, %false_val: tensor<?xf32>) -> tensor<?xf32> {
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
// CHECK:           %[[MEMREF:.*]] = bufferization.to_memref %[[TENSOR]] : memref<f32>
// CHECK:           %[[RESULT_MEMREF:.*]] = scf.for %[[VAL_6:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] iter_args(%[[ITER:.*]] = %[[MEMREF]]) -> (memref<f32>) {
// CHECK:             scf.yield %[[ITER]] : memref<f32>
// CHECK:           } {some_attr}
// CHECK:           %[[VAL_8:.*]] = bufferization.to_tensor %[[VAL_9:.*]] : memref<f32>
// CHECK:           return %[[VAL_8]] : tensor<f32>
// CHECK:         }
func.func @for(%arg0: tensor<f32>, %lb: index, %ub: index, %step: index) -> tensor<f32> {
  %ret = scf.for %iv = %lb to %ub step %step iter_args(%iter = %arg0) -> tensor<f32> {
    scf.yield %iter : tensor<f32>
  } {some_attr}
  return %ret : tensor<f32>
}

// Check whether this converts at all.
//
// It would previously fail altogether.
// CHECK-LABEL:   func @if_correct_recursive_legalization_behavior
// CHECK: "test.munge_tensor"
func.func @if_correct_recursive_legalization_behavior(%pred: i1, %tensor: tensor<f32>) -> tensor<f32> {
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
// CHECK:           %[[MEMREF:.*]] = bufferization.to_memref %[[TENSOR]] : memref<f32>
// CHECK:           %[[RESULT:.*]] = scf.for %[[IV:.*]] = %[[INDEX]] to %[[INDEX]] step %[[INDEX]] iter_args(%[[MEMREF_ITER:.*]] = %[[MEMREF]]) -> (memref<f32>) {
// CHECK:             %[[TENSOR_ITER:.*]] = bufferization.to_tensor %[[MEMREF_ITER]] : memref<f32>
// CHECK:             %[[TENSOR_MUNGED:.*]] = "test.munge_tensor"(%[[TENSOR_ITER]]) : (tensor<f32>) -> tensor<f32>
// CHECK:             %[[MEMREF_MUNGED:.*]] = bufferization.to_memref %[[TENSOR_MUNGED]] : memref<f32>
// CHECK:             scf.yield %[[MEMREF_MUNGED]] : memref<f32>
// CHECK:           }
// CHECK:           %[[TENSOR:.*]] = bufferization.to_tensor %[[RESULT:.*]] : memref<f32>
// CHECK:           return %[[TENSOR]] : tensor<f32>
// CHECK:         }
func.func @for_correct_recursive_legalization_behavior(%arg0: tensor<f32>, %index: index) -> tensor<f32> {
  %ret = scf.for %iv = %index to %index step %index iter_args(%iter = %arg0) -> tensor<f32> {
    %0 = "test.munge_tensor"(%iter) : (tensor<f32>) -> (tensor<f32>)
    scf.yield %0 : tensor<f32>
  }
  return %ret : tensor<f32>
}

// CHECK-LABEL:   func @bufferize_while(
// CHECK-SAME: %[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64, %[[ARG2:.*]]: tensor<f32>
// CHECK: %[[M:.*]] = bufferization.to_memref %[[ARG2]] : memref<f32>
// CHECK: %[[RES1:.*]]:3 = scf.while (%{{.*}} = %[[ARG0]], %{{.*}} = %[[M]]) : (i64, memref<f32>) -> (i64, i64, memref<f32>)
// CHECK: scf.condition(%{{.*}}) %{{.*}}, %{{.*}}, %{{.*}} : i64, i64, memref<f32>
// CHECK: ^bb0(%{{.*}}: i64, %{{.*}}: i64, %{{.*}}: memref<f32>):
// CHECK: scf.yield %{{.*}}, %{{.*}} : i64, memref<f32>
// CHECK:  %[[RES2:.*]] = bufferization.to_tensor %[[RES1]]#2 : memref<f32>
// CHECK:  return %[[RES1]]#1, %[[RES2]] : i64, tensor<f32>
func.func @bufferize_while(%arg0: i64, %arg1: i64, %arg2: tensor<f32>) -> (i64, tensor<f32>) {
  %c2_i64 = arith.constant 2 : i64
  %0:3 = scf.while (%arg3 = %arg0, %arg4 = %arg2) : (i64, tensor<f32>) -> (i64, i64, tensor<f32>) {
    %1 = arith.cmpi slt, %arg3, %arg1 : i64
    scf.condition(%1) %arg3, %arg3, %arg4 : i64, i64, tensor<f32>
  } do {
  ^bb0(%arg5: i64, %arg6: i64, %arg7: tensor<f32>):
    %1 = arith.muli %arg6, %c2_i64 : i64
    scf.yield %1, %arg7 : i64, tensor<f32>
  }
  return %0#1, %0#2 : i64, tensor<f32>
}
