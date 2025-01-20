// RUN: mlir-opt %s -pass-pipeline='builtin.module(func.func(test-affine-reify-value-bounds))' -verify-diagnostics \
// RUN:     -split-input-file | FileCheck %s

// CHECK-LABEL: func @memref_alloc(
//  CHECK-SAME:     %[[sz:.*]]: index
//       CHECK:   %[[c6:.*]] = arith.constant 6 : index
//       CHECK:   return %[[c6]], %[[sz]]
func.func @memref_alloc(%sz: index) -> (index, index) {
  %0 = memref.alloc(%sz) : memref<6x?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<6x?xf32>) -> (index)
  %2 = "test.reify_bound"(%0) {dim = 1} : (memref<6x?xf32>) -> (index)
  return %1, %2 : index, index
}

// -----

// CHECK-LABEL: func @memref_alloca(
//  CHECK-SAME:     %[[sz:.*]]: index
//       CHECK:   %[[c6:.*]] = arith.constant 6 : index
//       CHECK:   return %[[c6]], %[[sz]]
func.func @memref_alloca(%sz: index) -> (index, index) {
  %0 = memref.alloca(%sz) : memref<6x?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<6x?xf32>) -> (index)
  %2 = "test.reify_bound"(%0) {dim = 1} : (memref<6x?xf32>) -> (index)
  return %1, %2 : index, index
}

// -----

// CHECK-LABEL: func @memref_cast(
//       CHECK:   %[[c10:.*]] = arith.constant 10 : index
//       CHECK:   return %[[c10]]
func.func @memref_cast(%m: memref<10xf32>) -> index {
  %0 = memref.cast %m : memref<10xf32> to memref<?xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<?xf32>) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @memref_dim(
//  CHECK-SAME:     %[[m:.*]]: memref<?xf32>
//       CHECK:   %[[dim:.*]] = memref.dim %[[m]]
//       CHECK:   %[[dim:.*]] = memref.dim %[[m]]
//       CHECK:   return %[[dim]]
func.func @memref_dim(%m: memref<?xf32>) -> index {
  %c0 = arith.constant 0 : index
  %0 = memref.dim %m, %c0 : memref<?xf32>
  %1 = "test.reify_bound"(%0) : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @memref_get_global(
//       CHECK:   %[[c4:.*]] = arith.constant 4 : index
//       CHECK:   return %[[c4]]
memref.global "private" @gv0 : memref<4xf32> = dense<[0.0, 1.0, 2.0, 3.0]>
func.func @memref_get_global() -> index {
  %0 = memref.get_global @gv0 : memref<4xf32>
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<4xf32>) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @memref_rank(
//  CHECK-SAME:     %[[t:.*]]: memref<5xf32>
//       CHECK:   %[[c1:.*]] = arith.constant 1 : index
//       CHECK:   return %[[c1]]
func.func @memref_rank(%m: memref<5xf32>) -> index {
  %0 = memref.rank %m : memref<5xf32>
  %1 = "test.reify_bound"(%0) : (index) -> (index)
  return %1 : index
}

// -----

// CHECK-LABEL: func @memref_subview(
//  CHECK-SAME:     %[[m:.*]]: memref<?xf32>, %[[sz:.*]]: index
//       CHECK:   return %[[sz]]
func.func @memref_subview(%m: memref<?xf32>, %sz: index) -> index {
  %0 = memref.subview %m[2][%sz][1] : memref<?xf32> to memref<?xf32, strided<[1], offset: 2>>
  %1 = "test.reify_bound"(%0) {dim = 0} : (memref<?xf32, strided<[1], offset: 2>>) -> (index)
  return %1 : index
}
