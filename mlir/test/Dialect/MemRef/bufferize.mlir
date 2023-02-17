// RUN: mlir-opt -one-shot-bufferize %s | FileCheck %s

// CHECK-LABEL: func @tensor_store(
//  CHECK-SAME:     %[[t:.*]]: tensor<?xf32>, %[[m:.*]]: memref<?xf32>
//       CHECK:   %[[src:.*]] = bufferization.to_memref %[[t]]
//       CHECK:   memref.copy %[[src]], %[[m]]
//       CHECK:   return
func.func @tensor_store(%t: tensor<?xf32>, %m: memref<?xf32>) {
  memref.tensor_store %t, %m : memref<?xf32>
  return
}
