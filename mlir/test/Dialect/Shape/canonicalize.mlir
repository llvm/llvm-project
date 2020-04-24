// RUN: mlir-opt -split-input-file -canonicalize <%s | FileCheck %s --dump-input=fail

// -----
// CHECK-LABEL: func @f
func @f(%arg0: tensor<2x3x4xf32>) -> !shape.shape {
  // CHECK: shape.const_shape [2, 3, 4]
  %0 = "shape.shape_of"(%arg0) : (tensor<2x3x4xf32>) -> !shape.shape
  return %0 : !shape.shape
}

// -----
// Basic case.
// CHECK-LABEL: func @f
func @f() -> (!shape.shape, !shape.shape) {
  // CHECK: shape.const_shape [2, 3]
  // CHECK: shape.const_shape [4, 5]
  %c2 = constant 2 : i32
  %0 = "shape.const_shape"() {shape = dense<[2, 3, 4, 5]> : tensor<4xi64>} : () -> !shape.shape
  %head, %tail = "shape.split_at"(%0, %c2) : (!shape.shape, i32) -> (!shape.shape, !shape.shape)
  return %head, %tail : !shape.shape, !shape.shape

}

// -----
// Negative split point.
// CHECK-LABEL: func @f
func @f() -> (!shape.shape, !shape.shape) {
  // CHECK: shape.const_shape [2, 3, 4]
  // CHECK: shape.const_shape [5]
  %c-1 = constant -1 : i32
  %0 = "shape.const_shape"() {shape = dense<[2, 3, 4, 5]> : tensor<4xi64>} : () -> !shape.shape
  %head, %tail = "shape.split_at"(%0, %c-1) : (!shape.shape, i32) -> (!shape.shape, !shape.shape)
  return %head, %tail : !shape.shape, !shape.shape
}

// -----
// Out of range split point. No folding.
// CHECK-LABEL: func @f
func @f() -> (!shape.shape, !shape.shape) {
  // CHECK: shape.split_at
  %c5 = constant 5 : i32
  %0 = "shape.const_shape"() {shape = dense<[2, 3, 4, 5]> : tensor<4xi64>} : () -> !shape.shape
  %head, %tail = "shape.split_at"(%0, %c5) : (!shape.shape, i32) -> (!shape.shape, !shape.shape)
  return %head, %tail : !shape.shape, !shape.shape
}

// -----
// Basic case.
// CHECK-LABEL: func @f
func @f() -> !shape.shape {
  // CHECK: shape.const_shape [7, 2]
  %0 = shape.const_shape [1, 2]
  %1 = shape.const_shape [7, 1]
  %2 = "shape.broadcast"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  return %2 : !shape.shape
}

// -----
// Incompatible shapes. No folding.
// CHECK-LABEL: func @f
func @f() -> !shape.shape {
  // CHECK: shape.broadcast
  %0 = shape.const_shape [2]
  %1 = shape.const_shape [7]
  %2 = "shape.broadcast"(%0, %1) : (!shape.shape, !shape.shape) -> !shape.shape
  return %2 : !shape.shape
}

// -----
// Basic case.
// CHECK-LABEL: func @f
func @f() -> !shape.shape {
  // CHECK: shape.const_shape [0, 1, 2, 3]
  %lhs = shape.const_shape [0, 1]
  %rhs = shape.const_shape [2, 3]
  %0 = "shape.concat"(%lhs, %rhs) : (!shape.shape, !shape.shape) -> !shape.shape
  return %0 : !shape.shape
}

// -----
// Basic case.
// CHECK-LABEL: func @f
func @f() -> tensor<2xindex> {
  // CHECK: constant dense<[0, 1]> : tensor<2xindex>
  %cs = shape.const_shape [0, 1]
  %0 = "shape.to_extent_tensor"(%cs) : (!shape.shape) -> tensor<2xindex>
  return %0 : tensor<2xindex>
}
