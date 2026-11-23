// RUN: mlir-opt %s -one-shot-bufferize="bufferize-function-boundaries test-analysis-only" -split-input-file | FileCheck %s

func.func @not_elementwise(%a: tensor<5x6xf32>) -> tensor<5x6xf32> {
  %cst = arith.constant 5.0 : f32
  // CHECK: tensor.extract_slice
  // CHECK-SAME: {__inplace_operands_attr__ = ["false"]}
  %b = tensor.extract_slice %a[0, 0] [1, 6] [1, 1]
      : tensor<5x6xf32> to tensor<6xf32>
  // CHECK: linalg.generic
  // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]}
  %0 = linalg.generic 
    { iterator_types = ["parallel", "parallel"],
      indexing_maps = [ affine_map<(d0, d1) -> (d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>] }
    ins(%b: tensor<6xf32>) outs(%a: tensor<5x6xf32>) {
    ^bb0(%arg0: f32, %arg1: f32):
      %r = arith.addf %arg0, %arg1 : f32
      linalg.yield %r : f32
    } -> tensor<5x6xf32>
  return %0 : tensor<5x6xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>

// CHECK-LABEL: @elementwise_no_conflict_4
func.func @elementwise_no_conflict_4(%arg0: tensor<8x32x32x32xf32>, %arg1: tensor<32x32x32xf32>) -> tensor<8x32x32x32xf32> {
  %cst = arith.constant dense<3.000000e-02> : tensor<32x32x32xf32>
  %cst_0 = arith.constant dense<6.000000e-01> : tensor<32xf32>
  %cst_1 = arith.constant 0.000000e+00 : f32
  %r = scf.forall (%arg2, %arg3) in (8, 32) shared_outs(%arg4 = %arg0) -> (tensor<8x32x32x32xf32>) {
    // CHECK: tensor.extract_slice
    // CHECK-SAME: {__inplace_operands_attr__ = ["true", "none", "none"]}
    %extracted_slice = tensor.extract_slice %arg4[%arg2, %arg3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<8x32x32x32xf32> to tensor<32x32xf32>

    // CHECK: linalg.fill
    // CHECK-SAME: {__inplace_operands_attr__ = ["none", "true"]}
    %4 = linalg.fill ins(%cst_1 : f32) outs(%extracted_slice : tensor<32x32xf32>) -> tensor<32x32xf32>

    // CHECK: linalg.batch_reduce_matmul
    // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "true"]}
    %5 = linalg.batch_reduce_matmul ins(%arg1, %cst : tensor<32x32x32xf32>, tensor<32x32x32xf32>) outs(%4 : tensor<32x32xf32>) -> tensor<32x32xf32>

    // CHECK: linalg.generic
    // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "true"]}
    // %cst_0 has a non-identity layout may, but %5 and %extracted_slice still
    // bufferize to element-wise access.
    %6 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%5, %cst_0 : tensor<32x32xf32>, tensor<32xf32>) outs(%extracted_slice : tensor<32x32xf32>) {
    ^bb0(%in: f32, %in_4: f32, %out: f32):
      %8 = arith.addf %in, %in_4 : f32
      linalg.yield %8 : f32
    } -> tensor<32x32xf32>

    // CHECK: linalg.generic
    // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true"]}
    // They are different SSA values, but %6 and %extract_slice are equivalent.
    %7 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%6 : tensor<32x32xf32>) outs(%extracted_slice : tensor<32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      %8 = arith.maximumf %in, %cst_1 : f32
      linalg.yield %8 : f32
    } -> tensor<32x32xf32>
    scf.forall.in_parallel {
      // CHECK: tensor.parallel_insert_slice
      // CHECK-SAME: {__inplace_operands_attr__ = ["true", "true", "none", "none"]}
      tensor.parallel_insert_slice %7 into %arg4[%arg2, %arg3, 0, 0] [1, 1, 32, 32] [1, 1, 1, 1] : tensor<32x32xf32> into tensor<8x32x32x32xf32>
    }
  }
  return %r : tensor<8x32x32x32xf32>
}

// -----

// CHECK-LABEL: func @elementwise_access_regression(
//       CHECK:   linalg.fill {__inplace_operands_attr__ = ["none", "false"]}
//       CHECK:   linalg.map
//  CHECK-SAME:   {__inplace_operands_attr__ = ["true", "true", "true"]}
//       CHECK:   linalg.map
//  CHECK-SAME:   {__inplace_operands_attr__ = ["true", "true", "true"]}
func.func private @f(%arg: tensor<32x1xf32>) -> ()
func.func @elementwise_access_regression(%arg0: i32, %arg2: tensor<32x1xf32>, %arg3: tensor<32x1xf32>) {
      %cst_0 = arith.constant 0.000000e+00 : f32
      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %0 = tensor.empty() : tensor<32x1xf32>

      // This op must bufferize out-of-place so that the filled tensor is not
      // overwritten by the ops inside of the loop.
      %1 = linalg.fill ins(%cst_0 : f32) outs(%0 : tensor<32x1xf32>) -> tensor<32x1xf32>

      scf.for %arg1 = %c0_i32 to %arg0 step %c1_i32 : i32 {
        %2 = linalg.map { arith.subf } ins(%1, %arg2 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%0 : tensor<32x1xf32>)
        %3 = tensor.empty() : tensor<32x1xf32>
        %4 = linalg.map { arith.subf } ins(%2, %arg3 : tensor<32x1xf32>, tensor<32x1xf32>) outs(%3 : tensor<32x1xf32>)
        func.call @f(%4) : (tensor<32x1xf32>) -> ()
      }
      return
}

// -----

#bias = affine_map<(d0, d1) -> (d0)>
#transpose = affine_map<(d0, d1) -> (d1, d0)>

// The two uses of %dst have the same permutation map. The projected map of
// %bias does not participate in the conflict.
// CHECK-LABEL: func @matching_permutation_maps_bufferize_inplace(
//       CHECK:   linalg.generic
//  CHECK-SAME:   {__inplace_operands_attr__ = ["true", "true", "true"]}
func.func @matching_permutation_maps_bufferize_inplace(
    %bias: tensor<16xf32>,
    %dst: tensor<8x16xf32> {bufferization.writable = true})
    -> tensor<8x16xf32> {
  %0 = linalg.generic {
      indexing_maps = [#bias, #transpose, #transpose],
      iterator_types = ["parallel", "parallel"]}
      ins(%bias, %dst : tensor<16xf32>, tensor<8x16xf32>)
      outs(%dst : tensor<8x16xf32>) {
    ^bb0(%bias_elem: f32, %old: f32, %out: f32):
      %1 = arith.addf %old, %bias_elem : f32
      linalg.yield %1 : f32
    } -> tensor<8x16xf32>
  return %0 : tensor<8x16xf32>
}

// -----

#identity = affine_map<(d0, d1) -> (d0, d1)>
#transpose = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: func @mismatching_permutation_maps(
//       CHECK:   linalg.generic
//  CHECK-SAME:   {__inplace_operands_attr__ = ["true", "false"]}
func.func @mismatching_permutation_maps(
    %arg0: tensor<4x4xf32> {bufferization.writable = true})
    -> tensor<4x4xf32> {
  %0 = linalg.generic {
      indexing_maps = [#transpose, #identity],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<4x4xf32>) outs(%arg0 : tensor<4x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %out : f32
      linalg.yield %1 : f32
    } -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// -----

#identity = affine_map<(d0, d1) -> (d0, d1)>
#projected = affine_map<(d0, d1) -> (d1)>

// Matching maps are insufficient when they are not permutations. Because
// #projected drops d0, multiple iterations access the same element of %state.
// CHECK-LABEL: func @reject_matching_projected_maps(
//       CHECK:   linalg.generic
//  CHECK-SAME:   {__inplace_operands_attr__ = ["true", "true", "false"]}
func.func @reject_matching_projected_maps(
    %source: tensor<4x4xf32>,
    %state: tensor<4xf32> {bufferization.writable = true})
    -> tensor<4xf32> {
  %0 = linalg.generic {
      indexing_maps = [#identity, #projected, #projected],
      iterator_types = ["parallel", "parallel"]}
      ins(%source, %state : tensor<4x4xf32>, tensor<4xf32>)
      outs(%state : tensor<4xf32>) {
    ^bb0(%source_elem: f32, %state_elem: f32, %out: f32):
      %1 = arith.addf %source_elem, %state_elem : f32
      linalg.yield %1 : f32
    } -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// -----

#transpose = affine_map<(d0, d1) -> (d1, d0)>

// This is intentionally not a semantic reduction. The reduction iterator
// verifies that matching permutation maps are rejected unless all loops are
// parallel.
// CHECK-LABEL: func @reject_reduction_iterator_with_matching_permutation_maps(
//       CHECK:   linalg.generic
//  CHECK-SAME:   {__inplace_operands_attr__ = ["true", "false"]}
func.func @reject_reduction_iterator_with_matching_permutation_maps(
    %arg0: tensor<4x4xf32> {bufferization.writable = true})
    -> tensor<4x4xf32> {
  %0 = linalg.generic {
      indexing_maps = [#transpose, #transpose],
      iterator_types = ["parallel", "reduction"]}
      ins(%arg0 : tensor<4x4xf32>) outs(%arg0 : tensor<4x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %out : f32
      linalg.yield %1 : f32
    } -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// -----

#sparse = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>
#transpose = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: func @matching_permutation_maps_with_sparse_operand(
//       CHECK:   linalg.generic
//  CHECK-SAME:   {__inplace_operands_attr__ = ["true", "false"]}
func.func @matching_permutation_maps_with_sparse_operand(
    %arg0: tensor<4x4xf32, #sparse> {bufferization.writable = true})
    -> tensor<4x4xf32, #sparse> {
  %0 = linalg.generic {
      indexing_maps = [#transpose, #transpose],
      iterator_types = ["parallel", "parallel"]}
      ins(%arg0 : tensor<4x4xf32, #sparse>)
      outs(%arg0 : tensor<4x4xf32, #sparse>) {
    ^bb0(%in: f32, %out: f32):
      %1 = arith.addf %in, %out : f32
      linalg.yield %1 : f32
    } -> tensor<4x4xf32, #sparse>
  return %0 : tensor<4x4xf32, #sparse>
}
