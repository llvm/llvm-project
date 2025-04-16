// RUN: mlir-opt -int-range-optimizations --split-input-file %s | FileCheck %s

// CHECK-LABEL: @dim_const
// CHECK: %[[ret:.+]] = arith.constant 3 : index
// CHECK: return %[[ret]]
func.func @dim_const(%m: memref<3x5xi32>) -> index {
  %c0 = arith.constant 0 : index
  %0 = memref.dim %m, %c0 : memref<3x5xi32>
  return %0 : index
}

// -----

// CHECK-LABEL: @dim_any_static
// CHECK: %[[op:.+]] = memref.dim
// CHECK: %[[ret:.+]] = test.reflect_bounds {smax = 5 : index, smin = 3 : index, umax = 5 : index, umin = 3 : index} %[[op]]
// CHECK: return %[[ret]]
func.func @dim_any_static(%m: memref<3x5xi32>, %x: index) -> index {
  %0 = memref.dim %m, %x : memref<3x5xi32>
  %1 = test.reflect_bounds %0 : index
  return %1 : index
}

// -----

// CHECK-LABEL: @dim_dynamic
// CHECK: %[[op:.+]] = memref.dim
// CHECK: %[[ret:.+]] = test.reflect_bounds {smax = 9223372036854775807 : index, smin = 0 : index, umax = 9223372036854775807 : index, umin = 0 : index} %[[op]]
// CHECK: return %[[ret]]
func.func @dim_dynamic(%m: memref<?x5xi32>) -> index {
  %c0 = arith.constant 0 : index
  %0 = memref.dim %m, %c0 : memref<?x5xi32>
  %1 = test.reflect_bounds %0 : index
  return %1 : index
}

// -----

// CHECK-LABEL: @dim_any_dynamic
// CHECK: %[[op:.+]] = memref.dim
// CHECK: %[[ret:.+]] = test.reflect_bounds {smax = 9223372036854775807 : index, smin = 0 : index, umax = 9223372036854775807 : index, umin = 0 : index} %[[op]]
// CHECK: return %[[ret]]
func.func @dim_any_dynamic(%m: memref<?x5xi32>, %x: index) -> index {
  %0 = memref.dim %m, %x : memref<?x5xi32>
  %1 = test.reflect_bounds %0 : index
  return %1 : index
}

// -----

// CHECK-LABEL: @dim_some_omitting_dynamic
// CHECK: %[[op:.+]] = memref.dim
// CHECK: %[[ret:.+]] = test.reflect_bounds {smax = 5 : index, smin = 3 : index, umax = 5 : index, umin = 3 : index} %[[op]]
// CHECK: return %[[ret]]
func.func @dim_some_omitting_dynamic(%m: memref<?x3x5xi32>, %x: index) -> index {
  %c1 = arith.constant 1 : index
  %0 = arith.maxsi %x, %c1 : index
  %1 = memref.dim %m, %0 : memref<?x3x5xi32>
  %2 = test.reflect_bounds %1 : index
  return %2 : index
}

// -----

// CHECK-LABEL: @dim_unranked
// CHECK: %[[op:.+]] = memref.dim
// CHECK: %[[ret:.+]] = test.reflect_bounds {smax = 9223372036854775807 : index, smin = 0 : index, umax = 9223372036854775807 : index, umin = 0 : index} %[[op]]
// CHECK: return %[[ret]]
func.func @dim_unranked(%m: memref<*xi32>) -> index {
  %c0 = arith.constant 0 : index
  %0 = memref.dim %m, %c0 : memref<*xi32>
  %1 = test.reflect_bounds %0 : index
  return %1 : index
}
