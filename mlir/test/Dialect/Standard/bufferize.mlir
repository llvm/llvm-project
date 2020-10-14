// RUN: mlir-opt %s -std-bufferize | FileCheck %s

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
