// RUN: mlir-opt -int-range-optimizations --split-input-file %s | FileCheck %s

// CHECK-LABEL: @dim_const
// CHECK: %[[ret:.+]] = arith.constant 3 : index
// CHECK: return %[[ret]]
func.func @dim_const(%m: tensor<3x5xi32>) -> index {
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %m, %c0 : tensor<3x5xi32>
  return %0 : index
}

// -----

// CHECK-LABEL: @dim_any_static
// CHECK: %[[op:.+]] = tensor.dim
// CHECK: %[[ret:.+]] = test.reflect_bounds {smax = 5 : index, smin = 3 : index, umax = 5 : index, umin = 3 : index} %[[op]]
// CHECK: return %[[ret]]
func.func @dim_any_static(%m: tensor<3x5xi32>, %x: index) -> index {
  %0 = tensor.dim %m, %x : tensor<3x5xi32>
  %1 = test.reflect_bounds %0 : index
  return %1 : index
}

// -----

// CHECK-LABEL: @dim_dynamic
// CHECK: %[[op:.+]] = tensor.dim
// CHECK: %[[ret:.+]] = test.reflect_bounds {smax = 9223372036854775807 : index, smin = 0 : index, umax = 9223372036854775807 : index, umin = 0 : index} %[[op]]
// CHECK: return %[[ret]]
func.func @dim_dynamic(%m: tensor<?x5xi32>) -> index {
  %c0 = arith.constant 0 : index
  %0 = tensor.dim %m, %c0 : tensor<?x5xi32>
  %1 = test.reflect_bounds %0 : index
  return %1 : index
}

// -----

// CHECK-LABEL: @dim_any_dynamic
// CHECK: %[[op:.+]] = tensor.dim
// CHECK: %[[ret:.+]] = test.reflect_bounds {smax = 9223372036854775807 : index, smin = 0 : index, umax = 9223372036854775807 : index, umin = 0 : index} %[[op]]
// CHECK: return %[[ret]]
func.func @dim_any_dynamic(%m: tensor<?x5xi32>, %x: index) -> index {
  %0 = tensor.dim %m, %x : tensor<?x5xi32>
  %1 = test.reflect_bounds %0 : index
  return %1 : index
}

// -----

// CHECK-LABEL: @dim_some_omitting_dynamic
// CHECK: %[[op:.+]] = tensor.dim
// CHECK: %[[ret:.+]] = test.reflect_bounds {smax = 5 : index, smin = 3 : index, umax = 5 : index, umin = 3 : index} %[[op]]
// CHECK: return %[[ret]]
func.func @dim_some_omitting_dynamic(%m: tensor<?x3x5xi32>, %x: index) -> index {
  %c1 = arith.constant 1 : index
  %0 = arith.maxsi %x, %c1 : index
  %1 = tensor.dim %m, %0 : tensor<?x3x5xi32>
  %2 = test.reflect_bounds %1 : index
  return %2 : index
}
