// RUN: mlir-opt -transform-interpreter -cse -split-input-file -verify-diagnostics %s | FileCheck %s

// Test tiling tensor.concat on non-concat dimensions (dimension 0 tiled).

func.func @concat_tile_dim0(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x16xf32> {
  %0 = tensor.concat dim(1) %arg0, %arg1 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %concat = transform.structured.match ops{["tensor.concat"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    // Tile dimension 0 (non-concat), keep dimension 1 (concat) untiled.
    %tiled, %loop = transform.structured.tile_using_for %concat tile_sizes [2, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func.func @concat_tile_dim0(
// CHECK:         scf.for
// CHECK:           tensor.extract_slice
// CHECK:           tensor.extract_slice
// CHECK:           tensor.concat dim(1)
// CHECK:           tensor.insert_slice

// -----

// Test tiling tensor.concat with dynamic shapes (non-concat dimension is dynamic).

func.func @concat_tile_dynamic(%arg0: tensor<?x8xf32>, %arg1: tensor<?x8xf32>) -> tensor<?x16xf32> {
  %0 = tensor.concat dim(1) %arg0, %arg1 : (tensor<?x8xf32>, tensor<?x8xf32>) -> tensor<?x16xf32>
  return %0 : tensor<?x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %concat = transform.structured.match ops{["tensor.concat"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %concat tile_sizes [2, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func.func @concat_tile_dynamic(
// CHECK:         tensor.dim
// CHECK:         scf.for
// CHECK:           affine.min
// CHECK:           tensor.extract_slice
// CHECK:           tensor.extract_slice
// CHECK:           tensor.concat dim(1)
// CHECK:           tensor.insert_slice

// -----

// Test tiling tensor.concat when concat is on dim 0 and we tile dim 1.

func.func @concat_on_dim0_tile_dim1(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<8x8xf32> {
  %0 = tensor.concat dim(0) %arg0, %arg1 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<8x8xf32>
  return %0 : tensor<8x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %concat = transform.structured.match ops{["tensor.concat"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    // Tile only dimension 1 (non-concat), dimension 0 (concat) is untiled.
    %tiled, %loop = transform.structured.tile_using_for %concat tile_sizes [0, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func.func @concat_on_dim0_tile_dim1(
// CHECK:         scf.for
// CHECK:           tensor.extract_slice {{.*}}[0,
// CHECK:           tensor.extract_slice {{.*}}[0,
// CHECK:           tensor.concat dim(0)
// CHECK:           tensor.insert_slice

// -----

// Test tiling tensor.concat with 3 inputs.

func.func @concat_three_inputs(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x24xf32> {
  %0 = tensor.concat dim(1) %arg0, %arg1, %arg2 : (tensor<4x8xf32>, tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x24xf32>
  return %0 : tensor<4x24xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %concat = transform.structured.match ops{["tensor.concat"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %tiled, %loop = transform.structured.tile_using_for %concat tile_sizes [2, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func.func @concat_three_inputs(
// CHECK:         scf.for
// CHECK:           tensor.extract_slice
// CHECK:           tensor.extract_slice
// CHECK:           tensor.extract_slice
// CHECK:           tensor.concat dim(1)
// CHECK:           tensor.insert_slice

// -----

// Test 3D tensor concat tiling multiple dimensions.

func.func @concat_3d_tile_two_dims(%arg0: tensor<4x8x16xf32>, %arg1: tensor<4x8x16xf32>) -> tensor<4x8x32xf32> {
  %0 = tensor.concat dim(2) %arg0, %arg1 : (tensor<4x8x16xf32>, tensor<4x8x16xf32>) -> tensor<4x8x32xf32>
  return %0 : tensor<4x8x32xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %concat = transform.structured.match ops{["tensor.concat"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    // Tile dimensions 0 and 1 (non-concat), keep dimension 2 (concat) untiled.
    %tiled, %loop0, %loop1 = transform.structured.tile_using_for %concat tile_sizes [2, 4, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
// CHECK-LABEL: func.func @concat_3d_tile_two_dims(
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             tensor.extract_slice {{.*}} [2, 4, 16]
// CHECK:             tensor.extract_slice {{.*}} [2, 4, 16]
// CHECK:             tensor.concat dim(2)
// CHECK:             tensor.insert_slice

// -----

// Negative test: tiling the concat dimension should fail.

func.func @concat_tile_concat_dim_fail(%arg0: tensor<4x8xf32>, %arg1: tensor<4x8xf32>) -> tensor<4x16xf32> {
  // expected-error @below {{faild to tile operation}}
  // expected-error @below {{failed to generate tiling loops}}
  %0 = tensor.concat dim(1) %arg0, %arg1 : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %concat = transform.structured.match ops{["tensor.concat"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    // Attempt to tile the concat dimension (dim 1) - should fail.
    %tiled, %loop = transform.structured.tile_using_for %concat tile_sizes [0, 4]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}

// -----

// Negative test: tiling a dynamic concat dimension should fail.

func.func @concat_tile_dynamic_concat_dim_fail(%arg0: tensor<?x8xf32>, %arg1: tensor<?x8xf32>) -> tensor<?x8xf32> {
  // expected-error @below {{faild to tile operation}}
  // expected-error @below {{failed to generate tiling loops}}
  %0 = tensor.concat dim(0) %arg0, %arg1 : (tensor<?x8xf32>, tensor<?x8xf32>) -> tensor<?x8xf32>
  return %0 : tensor<?x8xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1 : !transform.any_op {transform.readonly}) {
    %concat = transform.structured.match ops{["tensor.concat"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    // Attempt to tile the dynamic concat dimension (dim 0) - should fail.
    %tiled, %loop = transform.structured.tile_using_for %concat tile_sizes [4, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.yield
  }
}
