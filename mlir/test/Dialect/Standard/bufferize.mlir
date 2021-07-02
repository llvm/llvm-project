// RUN: mlir-opt %s -std-bufferize | FileCheck %s

// CHECK-LABEL:   func @select(
// CHECK-SAME:                 %[[PRED:.*]]: i1,
// CHECK-SAME:                 %[[TRUE_VAL:.*]]: tensor<f32>,
// CHECK-SAME:                 %[[FALSE_VAL:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK:           %[[TRUE_VAL_MEMREF:.*]] = memref.buffer_cast %[[TRUE_VAL]] : memref<f32>
// CHECK:           %[[FALSE_VAL_MEMREF:.*]] = memref.buffer_cast %[[FALSE_VAL]] : memref<f32>
// CHECK:           %[[RET_MEMREF:.*]] = select %[[PRED]], %[[TRUE_VAL_MEMREF]], %[[FALSE_VAL_MEMREF]] : memref<f32>
// CHECK:           %[[RET:.*]] = memref.tensor_load %[[RET_MEMREF]] : memref<f32>
// CHECK:           return %[[RET]] : tensor<f32>
func @select(%arg0: i1, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<f32> {
  %0 = select %arg0, %arg1, %arg2 : tensor<f32>
  return %0 : tensor<f32>
}

// CHECK-LABEL:   func @index_cast(
// CHECK-SAME:  %[[TENSOR:.*]]: tensor<i32>, %[[SCALAR:.*]]: i32
func @index_cast(%tensor: tensor<i32>, %scalar: i32) -> (tensor<index>, index) {
  %index_tensor = index_cast %tensor : tensor<i32> to tensor<index>
  %index_scalar = index_cast %scalar : i32 to index
  return %index_tensor, %index_scalar : tensor<index>, index
}
// CHECK:  %[[MEMREF:.*]] = memref.buffer_cast %[[TENSOR]] : memref<i32>
// CHECK-NEXT: %[[INDEX_MEMREF:.*]] = index_cast %[[MEMREF]]
// CHECK-SAME:   memref<i32> to memref<index>
// CHECK-NEXT: %[[INDEX_TENSOR:.*]] = memref.tensor_load %[[INDEX_MEMREF]]
// CHECK: return %[[INDEX_TENSOR]]
