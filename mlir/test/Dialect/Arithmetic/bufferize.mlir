// RUN: mlir-opt %s -arith-bufferize | FileCheck %s

// CHECK-LABEL:   func @index_cast(
// CHECK-SAME:  %[[TENSOR:.*]]: tensor<i32>, %[[SCALAR:.*]]: i32
func @index_cast(%tensor: tensor<i32>, %scalar: i32) -> (tensor<index>, index) {
  %index_tensor = arith.index_cast %tensor : tensor<i32> to tensor<index>
  %index_scalar = arith.index_cast %scalar : i32 to index
  return %index_tensor, %index_scalar : tensor<index>, index
}
// CHECK:  %[[MEMREF:.*]] = memref.buffer_cast %[[TENSOR]] : memref<i32>
// CHECK-NEXT: %[[INDEX_MEMREF:.*]] = arith.index_cast %[[MEMREF]]
// CHECK-SAME:   memref<i32> to memref<index>
// CHECK-NEXT: %[[INDEX_TENSOR:.*]] = memref.tensor_load %[[INDEX_MEMREF]]
// CHECK: return %[[INDEX_TENSOR]]
