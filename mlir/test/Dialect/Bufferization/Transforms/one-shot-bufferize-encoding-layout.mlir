// Default: function-boundary-type-conversion=infer-layout-map.
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries=1 allow-unknown-ops" -split-input-file | FileCheck %s

// The tensor encoding implements `MemRefLayoutAttrInterface`, so it wins over
// the requested layout option and the resulting memref uses the encoding.
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries=1 allow-unknown-ops function-boundary-type-conversion=identity-layout-map unknown-type-conversion=identity-layout-map" -split-input-file | FileCheck %s
// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries=1 allow-unknown-ops function-boundary-type-conversion=fully-dynamic-layout-map unknown-type-conversion=fully-dynamic-layout-map" -split-input-file | FileCheck %s

// Exercises the `tensorEncodingToMemRefLayoutFn` hook on the three
// tensor-to-memref paths that do not delegate to an op-specific
// `BufferizableOpInterface::getBufferType`:
//   * function-boundary conversion (arg + result),
//   * `bufferization.alloc_tensor`,
//   * unknown-type fallback (unknown op result).

#transpose = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: func @encoding_layout_function_boundary(
//  CHECK-SAME:     %[[A:.*]]: memref<4x4xf32, #[[$MAP:[^>]+]]>) -> memref<4x4xf32, #[[$MAP]]> {
//       CHECK:   return %[[A]] : memref<4x4xf32, #[[$MAP]]>
func.func @encoding_layout_function_boundary(
    %arg0: tensor<4x4xf32, #transpose>) -> tensor<4x4xf32, #transpose> {
  return %arg0 : tensor<4x4xf32, #transpose>
}

// -----

#transpose = affine_map<(d0, d1) -> (d1, d0)>

// The alloc layout must come from the encoding, not from the default static
// identity layout used by `alloc_tensor` otherwise.
// CHECK-LABEL: func @encoding_layout_alloc_tensor(
//       CHECK:   %[[ALLOC:.*]] = memref.alloc() {{.*}} : memref<4x4xf32, #{{.*}}>
//       CHECK:   return %[[ALLOC]] : memref<4x4xf32, #{{.*}}>
func.func @encoding_layout_alloc_tensor() -> tensor<4x4xf32, #transpose> {
  %0 = bufferization.alloc_tensor() : tensor<4x4xf32, #transpose>
  return %0 : tensor<4x4xf32, #transpose>
}

// -----

#transpose = affine_map<(d0, d1) -> (d1, d0)>

// The unknown op stays on tensors but is bracketed by to_tensor/to_buffer
// conversions whose memref types must use the encoding layout.
// CHECK-LABEL: func @encoding_layout_unknown_op(
//  CHECK-SAME:     %[[A:.*]]: memref<4x4xf32, #[[$MAP:[^>]+]]>
//       CHECK:   %[[T:.*]] = bufferization.to_tensor %[[A]] : memref<4x4xf32, #[[$MAP]]>
//       CHECK:   %[[R:.*]] = "test.dummy_op"(%[[T]])
//       CHECK:   %[[B:.*]] = bufferization.to_buffer %[[R]] {{.*}} to memref<4x4xf32, #[[$MAP]]>
//       CHECK:   return %[[B]] : memref<4x4xf32, #[[$MAP]]>
func.func @encoding_layout_unknown_op(
    %arg0: tensor<4x4xf32, #transpose>) -> tensor<4x4xf32, #transpose> {
  %0 = "test.dummy_op"(%arg0)
      : (tensor<4x4xf32, #transpose>) -> tensor<4x4xf32, #transpose>
  return %0 : tensor<4x4xf32, #transpose>
}

// -----

// Control case: without an encoding that implements `MemRefLayoutAttrInterface`
// the default path is taken. The function boundary infers the layout (identity
// for an equivalent return), matching the behavior of other bufferization
// tests.
// CHECK-LABEL: func @no_encoding_function_boundary(
//  CHECK-SAME:     %[[A:.*]]: memref<4x4xf32{{.*}}>) -> memref<4x4xf32{{.*}}> {
//       CHECK:   return %[[A]]
func.func @no_encoding_function_boundary(
    %arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
  return %arg0 : tensor<4x4xf32>
}
