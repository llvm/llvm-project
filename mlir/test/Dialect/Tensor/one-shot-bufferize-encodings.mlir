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

// -----

func.func @concat_memory_space(%arg0: tensor<8xf32, 1>,
                               %arg1: tensor<8xf32, 1>) -> tensor<16xf32, 1> {
  %0 = tensor.concat dim(0) %arg0, %arg1
      : (tensor<8xf32, 1>, tensor<8xf32, 1>) -> tensor<16xf32, 1>
  return %0 : tensor<16xf32, 1>
}

// CHECK-LABEL: @concat_memory_space
//  CHECK-SAME: (%[[ARG0:.+]]: tensor<8xf32, 1 : i64>, %[[ARG1:.+]]: tensor<8xf32, 1 : i64>)
//   CHECK-DAG: %[[BUFFER0:.+]] = bufferization.to_buffer %[[ARG0]]
//   CHECK-DAG: %[[BUFFER1:.+]] = bufferization.to_buffer %[[ARG1]]
//       CHECK: %[[ALLOC:.+]] = memref.alloc() {{.*}} : memref<16xf32, 1>
//       CHECK: %[[SUBVIEW0:.+]] = memref.subview %[[ALLOC]][0] [8] [1]
//       CHECK: memref.copy %[[BUFFER0]], %[[SUBVIEW0]]
//       CHECK: %[[SUBVIEW1:.+]] = memref.subview %[[ALLOC]][8] [8] [1]
//       CHECK: memref.copy %[[BUFFER1]], %[[SUBVIEW1]]
//       CHECK: %[[RESULT:.+]] = bufferization.to_tensor %[[ALLOC]]
//       CHECK: return %[[RESULT]] : tensor<16xf32, 1 : i64>
