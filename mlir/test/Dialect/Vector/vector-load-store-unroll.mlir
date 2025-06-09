// RUN: mlir-opt %s -test-vector-load-store-unroll --split-input-file | FileCheck %s

// CHECK-LABEL: func.func @unroll_2D_vector_load(
// CHECK-SAME:  %[[ARG:.*]]: memref<4x4xf16>) -> vector<4x4xf16> {
func.func @unroll_2D_vector_load(%arg0: memref<4x4xf16>) -> vector<4x4xf16> {
  // CHECK: %[[C3:.*]] = arith.constant 3 : index
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<4x4xf16>
  // CHECK: %[[V0:.*]] = vector.load %[[ARG]][%[[C0]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
  // CHECK: %[[V1:.*]] = vector.insert_strided_slice %[[V0]], %[[CST]] {offsets = [0, 0], strides = [1]} : vector<4xf16> into vector<4x4xf16>
  // CHECK: %[[V2:.*]] = vector.load %[[ARG]][%[[C1]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
  // CHECK: %[[V3:.*]] = vector.insert_strided_slice %[[V2]], %[[V1]] {offsets = [1, 0], strides = [1]} : vector<4xf16> into vector<4x4xf16>
  // CHECK: %[[V4:.*]] = vector.load %[[ARG]][%[[C2]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
  // CHECK: %[[V5:.*]] = vector.insert_strided_slice %[[V4]], %[[V3]] {offsets = [2, 0], strides = [1]} : vector<4xf16> into vector<4x4xf16>
  // CHECK: %[[V6:.*]] = vector.load %[[ARG]][%[[C3]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
  // CHECK: %[[V7:.*]] = vector.insert_strided_slice %[[V6]], %[[V5]] {offsets = [3, 0], strides = [1]} : vector<4xf16> into vector<4x4xf16>
  // CHECK: return %[[V7]] : vector<4x4xf16>
  %c0 = arith.constant 0 : index
  %0 = vector.load %arg0[%c0, %c0] : memref<4x4xf16>, vector<4x4xf16>
  return %0 : vector<4x4xf16>
}

// CHECK-LABEL: func.func @unroll_2D_vector_store(
// CHECK-SAME:  %[[ARG0:.*]]: memref<4x4xf16>, %[[ARG1:.*]]: vector<4x4xf16>) {
func.func @unroll_2D_vector_store(%arg0: memref<4x4xf16>, %arg1: vector<4x4xf16>) {
  // CHECK: %[[C3:.*]] = arith.constant 3 : index
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[V0:.*]] = vector.extract %[[ARG1]][0] : vector<4xf16> from vector<4x4xf16>
  // CHECK: vector.store %[[V0]], %[[ARG0]][%[[C0]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
  // CHECK: %[[V1:.*]] = vector.extract %[[ARG1]][1] : vector<4xf16> from vector<4x4xf16>
  // CHECK: vector.store %[[V1]], %[[ARG0]][%[[C1]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
  // CHECK: %[[V2:.*]] = vector.extract %[[ARG1]][2] : vector<4xf16> from vector<4x4xf16>
  // CHECK: vector.store %[[V2]], %[[ARG0]][%[[C2]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
  // CHECK: %[[V3:.*]] = vector.extract %[[ARG1]][3] : vector<4xf16> from vector<4x4xf16>
  // CHECK: vector.store %[[V3]], %[[ARG0]][%[[C3]], %[[C0]]] : memref<4x4xf16>, vector<4xf16>
  %c0 = arith.constant 0 : index
  vector.store %arg1, %arg0[%c0, %c0] : memref<4x4xf16>, vector<4x4xf16>
  return
}

// CHECK-LABEL: func.func @unroll_vector_load(
// CHECK-SAME:  %[[ARG:.*]]: memref<4x4x4x4xf16>) -> vector<2x2xf16> {
func.func @unroll_vector_load(%arg0: memref<4x4x4x4xf16>) -> vector<2x2xf16> {
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<2x2xf16>
  // CHECK: %[[V0:.*]] = vector.load %[[ARG]][%[[C1]], %[[C1]], %[[C1]], %[[C1]]] : memref<4x4x4x4xf16>, vector<2xf16>
  // CHECK: %[[V1:.*]] = vector.insert_strided_slice %[[V0]], %[[CST]] {offsets = [0, 0], strides = [1]} : vector<2xf16> into vector<2x2xf16>
  // CHECK: %[[V2:.*]] = vector.load %[[ARG]][%[[C1]], %[[C1]], %[[C2]], %[[C1]]] : memref<4x4x4x4xf16>, vector<2xf16>
  // CHECK: %[[V3:.*]] = vector.insert_strided_slice %[[V2]], %[[V1]] {offsets = [1, 0], strides = [1]} : vector<2xf16> into vector<2x2xf16>
  // CHECK: return %[[V3]] : vector<2x2xf16>
  %c1 = arith.constant 1 : index
  %0 = vector.load %arg0[%c1, %c1, %c1, %c1] : memref<4x4x4x4xf16>, vector<2x2xf16>
  return %0 : vector<2x2xf16>
}

// CHECK-LABEL: func.func @unroll_vector_store(
// CHECK-SAME:  %[[ARG0:.*]]: memref<4x4x4x4xf16>, %[[ARG1:.*]]: vector<2x2xf16>) {
func.func @unroll_vector_store(%arg0: memref<4x4x4x4xf16>, %arg1: vector<2x2xf16>) {
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[V0:.*]] = vector.extract %[[ARG1]][0] : vector<2xf16> from vector<2x2xf16>
  // CHECK: vector.store %[[V0]], %[[ARG0]][%[[C1]], %[[C1]], %[[C1]], %[[C1]]] : memref<4x4x4x4xf16>, vector<2xf16>
  // CHECK: %[[V1:.*]] = vector.extract %[[ARG1]][1] : vector<2xf16> from vector<2x2xf16>
  // CHECK: vector.store %[[V1]], %[[ARG0]][%[[C1]], %[[C1]], %[[C2]], %[[C1]]] : memref<4x4x4x4xf16>, vector<2xf16>
  %c1 = arith.constant 1 : index
  vector.store %arg1, %arg0[%c1, %c1, %c1, %c1] : memref<4x4x4x4xf16>, vector<2x2xf16>
  return
}
