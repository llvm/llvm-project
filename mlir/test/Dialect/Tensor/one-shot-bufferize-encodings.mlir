// RUN: mlir-opt %s -one-shot-bufferize="use-encoding-for-memory-space" -split-input-file | FileCheck %s

func.func @from_elements(%fill: f32, %f: f32, %idx: index) -> tensor<3xf32, 1> {
  %t = tensor.from_elements %fill, %fill, %fill : tensor<3xf32, 1>
  %i = tensor.insert %f into %t[%idx] : tensor<3xf32, 1>
  return %i : tensor<3xf32, 1>
}

// CHECK-LABEL: @from_elements
//  CHECK-SAME: (%[[arg0:.+]]: f32, %[[arg1:.+]]: f32, %[[arg2:.+]]: index) -> tensor<3xf32, 1 : i64>
//       CHECK:     %[[alloc:.+]] = memref.alloc() {{.*}} : memref<3xf32, 1>
//       CHECK-DAG:     %[[c0:.+]] = arith.constant 0 : index
//       CHECK-DAG:     %[[c1:.+]] = arith.constant 1 : index
//       CHECK-DAG:     %[[c2:.+]] = arith.constant 2 : index
//       CHECK:     memref.store %[[arg0]], %[[alloc]][%[[c0]]] : memref<3xf32, 1>
//       CHECK:     memref.store %[[arg0]], %[[alloc]][%[[c1]]] : memref<3xf32, 1>
//       CHECK:     memref.store %[[arg0]], %[[alloc]][%[[c2]]] : memref<3xf32, 1>
//       CHECK:     memref.store %[[arg1]], %[[alloc]][%[[arg2]]] : memref<3xf32, 1>
//       CHECK:     %[[v0:.+]] = bufferization.to_tensor %[[alloc]] : memref<3xf32, 1> to tensor<3xf32, 1 : i64>
//       CHECK:     return %[[v0]] : tensor<3xf32, 1 : i64>
