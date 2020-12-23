// RUN: mlir-opt %s -tensor-bufferize | FileCheck %s

// CHECK-LABEL:   func @tensor.cast(
// CHECK-SAME:                      %[[TENSOR:.*]]: tensor<?xindex>) -> tensor<2xindex> {
// CHECK:           %[[MEMREF:.*]] = tensor_to_memref %[[TENSOR]]
// CHECK:           %[[CASTED:.*]] = memref_cast %[[MEMREF]] : memref<?xindex> to memref<2xindex>
// CHECK:           %[[RET:.*]] = tensor_load %[[CASTED]]
// CHECK:           return %[[RET]] : tensor<2xindex>
func @tensor.cast(%arg0: tensor<?xindex>) -> tensor<2xindex> {
  %0 = tensor.cast %arg0 : tensor<?xindex> to tensor<2xindex>
  return %0 : tensor<2xindex>
}

// CHECK-LABEL:   func @tensor.cast_from_unranked(
// CHECK-SAME:                                    %[[TENSOR:.*]]: tensor<*xf32>) -> tensor<2xf32> {
// CHECK:           %[[MEMREF:.*]] = tensor_to_memref %[[TENSOR]] : memref<*xf32>
// CHECK:           %[[CASTED_MEMREF:.*]] = memref_cast %[[MEMREF]] : memref<*xf32> to memref<2xf32>
// CHECK:           %[[RET:.*]] = tensor_load %[[CASTED_MEMREF]] : memref<2xf32>
// CHECK:           return %[[RET]] : tensor<2xf32>
func @tensor.cast_from_unranked(%arg0: tensor<*xf32>) -> tensor<2xf32> {
  %0 = tensor.cast %arg0 : tensor<*xf32> to tensor<2xf32>
  return %0 : tensor<2xf32>
}

// CHECK-LABEL:   func @tensor.cast_to_unranked(
// CHECK-SAME:                                  %[[TENSOR:.*]]: tensor<2xf32>) -> tensor<*xf32> {
// CHECK:           %[[MEMREF:.*]] = tensor_to_memref %[[TENSOR]] : memref<2xf32>
// CHECK:           %[[CASTED_MEMREF:.*]] = memref_cast %[[MEMREF]] : memref<2xf32> to memref<*xf32>
// CHECK:           %[[RET:.*]] = tensor_load %[[CASTED_MEMREF]] : memref<*xf32>
// CHECK:           return %[[RET]] : tensor<*xf32>
func @tensor.cast_to_unranked(%arg0: tensor<2xf32>) -> tensor<*xf32> {
  %0 = tensor.cast %arg0 : tensor<2xf32> to tensor<*xf32>
  return %0 : tensor<*xf32>
}

// CHECK-LABEL:   func @extract(
// CHECK-SAME:                  %[[TENSOR:.*]]: tensor<?xf32>,
// CHECK-SAME:                  %[[IDX:.*]]: index) -> f32 {
// CHECK:           %[[MEMREF:.*]] = tensor_to_memref %[[TENSOR]] : memref<?xf32>
// CHECK:           %[[RET:.*]] = load %[[MEMREF]][%[[IDX]]] : memref<?xf32>
// CHECK:           return %[[RET]] : f32
// CHECK:         }
func @extract(%arg0: tensor<?xf32>, %arg1: index) -> f32 {
  %0 = tensor.extract %arg0[%arg1] : tensor<?xf32>
  return %0 : f32
}
