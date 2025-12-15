// RUN: mlir-opt -resolve-shaped-type-result-dims -split-input-file %s | FileCheck %s

func.func @fat_raw_buffer_cast_static_dim(%arg0: memref<2x3xf32>) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cast = amdgpu.fat_raw_buffer_cast %arg0 : memref<2x3xf32>
      to memref<2x3xf32, #amdgpu.address_space<fat_raw_buffer>>
  %d0 = memref.dim %cast, %c0 : memref<2x3xf32, #amdgpu.address_space<fat_raw_buffer>>
  %d1 = memref.dim %cast, %c1 : memref<2x3xf32, #amdgpu.address_space<fat_raw_buffer>>
  return %d0, %d1 : index, index
}
//      CHECK: func @fat_raw_buffer_cast_static_dim
//  CHECK-DAG:   %[[C2:.+]] = arith.constant 2 : index
//  CHECK-DAG:   %[[C3:.+]] = arith.constant 3 : index
//      CHECK:   return %[[C2]], %[[C3]]

// -----

func.func @fat_raw_buffer_cast_dynamic_dim(%arg0: memref<4x?xf32>) -> (index, index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cast = amdgpu.fat_raw_buffer_cast %arg0 : memref<4x?xf32>
      to memref<4x?xf32, #amdgpu.address_space<fat_raw_buffer>>
  %d0 = memref.dim %cast, %c0 : memref<4x?xf32, #amdgpu.address_space<fat_raw_buffer>>
  %d1 = memref.dim %cast, %c1 : memref<4x?xf32, #amdgpu.address_space<fat_raw_buffer>>
  return %d0, %d1 : index, index
}
//      CHECK: func @fat_raw_buffer_cast_dynamic_dim
// CHECK-SAME:   %[[ARG0:[a-zA-Z0-9_]+]]: memref<4x?xf32>
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C4:.+]] = arith.constant 4 : index
//      CHECK:   %[[D1:.+]] = memref.dim %[[ARG0]], %[[C1]]
//      CHECK:   return %[[C4]], %[[D1]]
