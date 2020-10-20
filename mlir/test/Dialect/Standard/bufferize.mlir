// RUN: mlir-opt %s -std-bufferize | FileCheck %s

// CHECK-LABEL:   func @extract_element(
// CHECK-SAME:                          %[[TENSOR:.*]]: tensor<?xf32>,
// CHECK-SAME:                          %[[IDX:.*]]: index) -> f32 {
// CHECK:           %[[MEMREF:.*]] = tensor_to_memref %[[TENSOR]] : memref<?xf32>
// CHECK:           %[[RET:.*]] = load %[[MEMREF]][%[[IDX]]] : memref<?xf32>
// CHECK:           return %[[RET]] : f32
// CHECK:         }
func @extract_element(%arg0: tensor<?xf32>, %arg1: index) -> f32 {
  %0 = extract_element %arg0[%arg1] : tensor<?xf32>
  return %0 : f32
}

// CHECK-LABEL:   func @tensor_cast(
// CHECK-SAME:                      %[[TENSOR:.*]]: tensor<?xindex>) -> tensor<2xindex> {
// CHECK:           %[[MEMREF:.*]] = tensor_to_memref %[[TENSOR]]
// CHECK:           %[[CASTED:.*]] = memref_cast %[[MEMREF]] : memref<?xindex> to memref<2xindex>
// CHECK:           %[[RET:.*]] = tensor_load %[[CASTED]]
// CHECK:           return %[[RET]] : tensor<2xindex>
func @tensor_cast(%arg0: tensor<?xindex>) -> tensor<2xindex> {
  %0 = tensor_cast %arg0 : tensor<?xindex> to tensor<2xindex>
  return %0 : tensor<2xindex>
}

// CHECK-LABEL:   func @tensor_from_elements(
// CHECK-SAME:                               %[[ELEM0:.*]]: index,
// CHECK-SAME:                               %[[ELEM1:.*]]: index) -> tensor<2xindex> {
// CHECK:           %[[MEMREF:.*]] = alloc()
// CHECK:           %[[C0:.*]] = constant 0 : index
// CHECK:           store %[[ELEM0]], %[[MEMREF]][%[[C0]]]
// CHECK:           %[[C1:.*]] = constant 1 : index
// CHECK:           store %[[ELEM1]], %[[MEMREF]][%[[C1]]]
// CHECK:           %[[RET:.*]] = tensor_load %[[MEMREF]]
// CHECK:           return %[[RET]] : tensor<2xindex>
func @tensor_from_elements(%arg0: index, %arg1: index) -> tensor<2xindex> {
  %0 = tensor_from_elements %arg0, %arg1 : tensor<2xindex>
  return %0 : tensor<2xindex>
}
