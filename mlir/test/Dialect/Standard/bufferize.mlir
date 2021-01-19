// RUN: mlir-opt %s -std-bufferize | FileCheck %s

// CHECK-LABEL:   func @dim(
// CHECK-SAME:              %[[TENSOR:.*]]: tensor<f32>,
// CHECK-SAME:              %[[INDEX:.*]]: index) -> index {
// CHECK:           %[[MEMREF:.*]] = tensor_to_memref %[[TENSOR]] : memref<f32>
// CHECK:           %[[EXTENT:.*]] = dim %[[MEMREF]], %[[INDEX]] : memref<f32>
// CHECK:           return %[[EXTENT]] : index
func @dim(%arg0: tensor<f32>, %arg1: index) -> index {
  %0 = dim %arg0, %arg1 : tensor<f32>
  return %0 : index
}

// CHECK-LABEL:   func @select(
// CHECK-SAME:                 %[[PRED:.*]]: i1,
// CHECK-SAME:                 %[[TRUE_VAL:.*]]: tensor<f32>,
// CHECK-SAME:                 %[[FALSE_VAL:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK:           %[[TRUE_VAL_MEMREF:.*]] = tensor_to_memref %[[TRUE_VAL]] : memref<f32>
// CHECK:           %[[FALSE_VAL_MEMREF:.*]] = tensor_to_memref %[[FALSE_VAL]] : memref<f32>
// CHECK:           %[[RET_MEMREF:.*]] = select %[[PRED]], %[[TRUE_VAL_MEMREF]], %[[FALSE_VAL_MEMREF]] : memref<f32>
// CHECK:           %[[RET:.*]] = tensor_load %[[RET_MEMREF]] : memref<f32>
// CHECK:           return %[[RET]] : tensor<f32>
func @select(%arg0: i1, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  %0 = select %arg0, %arg1, %arg2 : tensor<f32>
  return %0 : tensor<f32>
}
