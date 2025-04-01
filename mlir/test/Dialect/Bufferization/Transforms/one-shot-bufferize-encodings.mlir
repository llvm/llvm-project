// RUN: mlir-opt %s -one-shot-bufferize="use-encoding-for-memory-space" -split-input-file | FileCheck %s

func.func @alloc_tesor_with_space_no_encoding() -> tensor<128xf32> {
  %0 = bufferization.alloc_tensor() {memory_space = 1 : i64} : tensor<128xf32>
  return %0 : tensor<128xf32>
}

// CHECK-LABEL: @alloc_tesor_with_space_no_encoding
//  CHECK-SAME: () -> tensor<128xf32> {
//       CHECK:     %[[alloc:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128xf32, 1>
//       CHECK:     %[[v0:.+]] = bufferization.to_tensor %[[alloc]] : memref<128xf32, 1> to tensor<128xf32>
//       CHECK:     return %[[v0]] : tensor<128xf32>

// -----

func.func @alloc_tesor_with_space_and_cast() -> tensor<128xf32, 1> {
  %0 = bufferization.alloc_tensor() {memory_space = 1 : i64} : tensor<128xf32>
  %1 = tensor.cast %0 : tensor<128xf32> to tensor<128xf32, 1>
  return %1 : tensor<128xf32, 1>
}

// CHECK-LABEL: @alloc_tesor_with_space_and_cast
//  CHECK-SAME: () -> tensor<128xf32, 1 : i64> {
//       CHECK:     %[[alloc:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128xf32, 1>
//       CHECK:     %[[v0:.+]] = bufferization.to_tensor %[[alloc]] : memref<128xf32, 1> to tensor<128xf32, 1 : i64>
//       CHECK:     return %[[v0]] : tensor<128xf32, 1 : i64>

// -----

func.func @alloc_tesor_with_space_with_encoding() -> tensor<128xf32, 1 : i64> {
  %0 = bufferization.alloc_tensor() {memory_space = 1 : i64} : tensor<128xf32, 1 : i64>
  return %0 : tensor<128xf32, 1 : i64>
}

// CHECK-LABEL: @alloc_tesor_with_space_with_encoding
//  CHECK-SAME: () -> tensor<128xf32, 1 : i64> {
//       CHECK:     %[[alloc:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128xf32, 1>
//       CHECK:     %[[v0:.+]] = bufferization.to_tensor %[[alloc]] : memref<128xf32, 1> to tensor<128xf32, 1 : i64>
//       CHECK:     return %[[v0]] : tensor<128xf32, 1 : i64>

// -----

func.func @alloc_tesor_copy_from_default_space(%arg0: tensor<128xf32>) -> tensor<128xf32> {
  %0 = bufferization.alloc_tensor() copy(%arg0) {memory_space = 1 : i64} : tensor<128xf32>
  return %0 : tensor<128xf32>
}

// CHECK-LABEL: @alloc_tesor_copy_from_default_space
//  CHECK-SAME: (%[[arg0:.+]]: tensor<128xf32>) -> tensor<128xf32> {
//       CHECK:     %[[v0:.+]] = bufferization.to_memref %[[arg0]] : tensor<128xf32> to memref<128xf32, strided<[?], offset: ?>>
//       CHECK:     %[[alloc:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128xf32, 1>
//       CHECK:     memref.copy %[[v0]], %[[alloc]] : memref<128xf32, strided<[?], offset: ?>> to memref<128xf32, 1>
//       CHECK:     %[[v1:.+]] = bufferization.to_tensor %[[alloc]] : memref<128xf32, 1> to tensor<128xf32>
//       CHECK:     return %[[v1]] : tensor<128xf32>

// -----

func.func @alloc_tesor_copy_from_non_default_space(%arg0: tensor<128xf32, 1>) -> tensor<128xf32, 2> {
  %0 = bufferization.alloc_tensor() copy(%arg0) {memory_space = 2 : i64} : tensor<128xf32, 1>
  %1 = tensor.cast %0 : tensor<128xf32, 1> to tensor<128xf32, 2>
  return %1 : tensor<128xf32, 2>
}

// CHECK-LABEL: @alloc_tesor_copy_from_non_default_space
//  CHECK-SAME: (%[[arg0:.+]]: tensor<128xf32, 1 : i64>) -> tensor<128xf32, 2 : i64> {
//       CHECK:     %[[v0:.+]] = bufferization.to_memref %[[arg0]] : tensor<128xf32, 1 : i64> to memref<128xf32, strided<[?], offset: ?>, 1>
//       CHECK:     %[[alloc:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128xf32, 2>
//       CHECK:     memref.copy %[[v0]], %[[alloc]] : memref<128xf32, strided<[?], offset: ?>, 1> to memref<128xf32, 2>
//       CHECK:     %[[v1:.+]] = bufferization.to_tensor %[[alloc]] : memref<128xf32, 2> to tensor<128xf32, 2 : i64>
//       CHECK:     return %[[v1]] : tensor<128xf32, 2 : i64>

// -----

// TODO: this should be illegal since ultimately we can not eliminate the `bufferization.to_tensor` when we
// bufferize function boundaries.
func.func @alloc_tesor_copy_from_non_default_space_no_cast(%arg0: tensor<128xf32, 1>,
                                                           %arg1: tensor<4xf32, 1>) -> tensor<128xf32, 1> {
  %0 = bufferization.alloc_tensor() copy(%arg0) {memory_space = 2 : i64} : tensor<128xf32, 1>
  %1 = tensor.insert_slice %arg1 into %arg0 [0][4][1] : tensor<4xf32, 1> into tensor<128xf32, 1>
  return %0 : tensor<128xf32, 1>
}

// CHECK-LABEL: @alloc_tesor_copy_from_non_default_space_no_cast
//  CHECK-SAME: (%[[arg0:.+]]: tensor<128xf32, 1 : i64>, %[[arg1:.+]]: tensor<4xf32, 1 : i64>) -> tensor<128xf32, 1 : i64> {
//       CHECK:     %[[v0:.+]] = bufferization.to_memref %[[arg1]] : tensor<4xf32, 1 : i64> to memref<4xf32, strided<[?], offset: ?>, 1>
//       CHECK:     %[[v1:.+]] = bufferization.to_memref %[[arg0]] : tensor<128xf32, 1 : i64> to memref<128xf32, strided<[?], offset: ?>, 1>
//       CHECK:     %[[v2:.+]] = bufferization.to_memref %[[arg0]] : tensor<128xf32, 1 : i64> to memref<128xf32, strided<[?], offset: ?>, 1>
//       CHECK:     %[[alloc:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128xf32, 2>
//       CHECK:     memref.copy %[[v2]], %[[alloc]] : memref<128xf32, strided<[?], offset: ?>, 1> to memref<128xf32, 2>
//       CHECK:     %[[v3:.+]] = bufferization.to_tensor %[[alloc]] : memref<128xf32, 2> to tensor<128xf32, 1 : i64>
//       CHECK:     %[[alloc_0:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128xf32, 1>
//       CHECK:     memref.copy %[[v1]], %[[alloc_0]] : memref<128xf32, strided<[?], offset: ?>, 1> to memref<128xf32, 1>
//       CHECK:     %[[subview:.+]] = memref.subview %[[alloc_0]][0] [4] [1] : memref<128xf32, 1> to memref<4xf32, strided<[1]>, 1>
//       CHECK:     memref.copy %[[v0]], %[[subview]] : memref<4xf32, strided<[?], offset: ?>, 1> to memref<4xf32, strided<[1]>, 1>
//       CHECK:     return %[[v3]] : tensor<128xf32, 1 : i64>

// -----

func.func @materialize_in_destination(%arg0: tensor<128xf32, 1>) -> tensor<128xf32, 2> {
  %0 = bufferization.alloc_tensor () {memory_space = 2 : i64} : tensor<128xf32, 2>
  %1 = bufferization.materialize_in_destination %arg0 in %0 : (tensor<128xf32, 1>, tensor<128xf32, 2>) -> tensor<128xf32, 2>
  return %1 : tensor<128xf32, 2>
}

// CHECK-LABEL: @materialize_in_destination
//  CHECK-SAME: (%[[arg0:.+]]: tensor<128xf32, 1 : i64>) -> tensor<128xf32, 2 : i64> {
//       CHECK:     %[[v0:.+]] = bufferization.to_memref %[[arg0]] : tensor<128xf32, 1 : i64> to memref<128xf32, strided<[?], offset: ?>, 1>
//       CHECK:     %[[alloc:.+]] = memref.alloc() {alignment = 64 : i64} : memref<128xf32, 2>
//       CHECK:     memref.copy %[[v0]], %[[alloc]] : memref<128xf32, strided<[?], offset: ?>, 1> to memref<128xf32, 2>
//       CHECK:     %[[v1:.+]] = bufferization.to_tensor %[[alloc]] : memref<128xf32, 2> to tensor<128xf32, 2 : i64>
//       CHECK:     return %[[v1]] : tensor<128xf32, 2 : i64>
