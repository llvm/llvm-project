// RUN: mlir-opt %s -tensor-bufferize | FileCheck %s

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
