// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

///----------------------------------------------------------------------------------------
/// Tests for tensor.pad
///----------------------------------------------------------------------------------------

// CHECK-LABEL: func @test_masked_vectorize_pad
func.func @test_masked_vectorize_pad(
  %0 : tensor<?x?xf32>, %h0 : index, %h1 : index)
    -> tensor<2x4xf32>
{
  //  CHECK-DAG: %[[c42:.*]] = arith.constant 4.243000e+01 : f32
  //  CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  //  CHECK-DAG: %[[c0_0:.*]] = arith.constant 0 : index
  //      CHECK: %[[d0:.*]] = tensor.dim {{.*}} : tensor<?x?xf32>
  //      CHECK: %[[d1:.*]] = tensor.dim {{.*}} : tensor<?x?xf32>
  //      CHECK: %[[mask:.*]] = vector.create_mask %[[d0]], %[[d1]] : vector<2x4xi1>
  //      CHECK: %[[masked_read:.*]] = vector.mask %[[mask]] {
  // CHECK-SAME:   vector.transfer_read %{{.*}}[%[[c0_0]], %[[c0_0]]], %[[c42]]
  // CHECK-SAME:   {in_bounds = [true, true]} : tensor<?x?xf32>, vector<2x4xf32>
  // CHECK-SAME: } : vector<2x4xi1> -> vector<2x4xf32>
  //  CHECK-DAG: %[[c0_1:.*]] = arith.constant 0 : index
  //  CHECK-DAG: %[[empty:.*]] = tensor.empty() : tensor<2x4xf32>
  //      CHECK: vector.transfer_write %[[masked_read]], %[[empty]][%[[c0_1]], %[[c0_1]]]
  // CHECK-SAME:   {in_bounds = [true, true]} : vector<2x4xf32>, tensor<2x4xf32>
  %cst = arith.constant 42.43 : f32
  %c0 = arith.constant 0 : index
  %1 = tensor.pad %0 low[0, %c0] high[%h0, %h1]  {
    ^bb0(%hh1: index, %hh2: index):
      tensor.yield %cst : f32
    } : tensor<?x?xf32> to tensor<2x4xf32>
  return %1: tensor<2x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [2, 4] : !transform.any_op
    transform.yield
  }
}

// -----

//       CHECK: #[[MAP:.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
//       CHECK: func @test_masked_vectorize_dynamic_pad
func.func @test_masked_vectorize_dynamic_pad(
  %0 : tensor<?x?xf32>, %h0 : index, %h1 : index)
    -> tensor<?x?xf32>
{
  //  CHECK-DAG: %[[c42:.*]] = arith.constant 4.243000e+01 : f32
  //  CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
  //  CHECK-DAG: %[[res_d0:.+]] = affine.apply #[[MAP]]()
  //  CHECK-DAG: %[[res_d1:.+]] = affine.apply #[[MAP]]()
  //      CHECK: %[[c0_2:.*]] = arith.constant 0 : index
  //      CHECK: %[[d0:.*]] = tensor.dim {{.*}} : tensor<?x?xf32>
  //      CHECK: %[[d1:.*]] = tensor.dim {{.*}} : tensor<?x?xf32>
  //      CHECK: %[[mask:.*]] = vector.create_mask %[[d0]], %[[d1]] : vector<2x4xi1>
  //      CHECK: %[[masked_read:.*]] = vector.mask %[[mask]] {
  // CHECK-SAME:   vector.transfer_read %{{.*}}[%[[c0_2]], %[[c0_2]]], %[[c42]]
  // CHECK-SAME:   {in_bounds = [true, true]} : tensor<?x?xf32>, vector<2x4xf32>
  // CHECK-SAME: } : vector<2x4xi1> -> vector<2x4xf32>
  //  CHECK-DAG: %[[empty:.*]] = tensor.empty(%[[res_d0]], %[[res_d1]]) : tensor<?x?xf32>
  //  CHECK-DAG: %[[c0_3:.*]] = arith.constant 0 : index
  //  CHECK-DAG: %[[d2:.*]] = tensor.dim %[[empty]], {{.*}} : tensor<?x?xf32>
  //  CHECK-DAG: %[[d3:.*]] = tensor.dim %[[empty]], {{.*}} : tensor<?x?xf32>
  //      CHECK: %[[mask_2:.*]] = vector.create_mask %[[d2]], %[[d3]] : vector<2x4xi1>
  //      CHECK: %[[masked_write:.*]] = vector.mask %[[mask_2]] {
  // CHECK-SAME: vector.transfer_write %[[masked_read]], %[[empty]][%[[c0_3]], %[[c0_3]]]
  // CHECK-SAME:   {in_bounds = [true, true]} : vector<2x4xf32>, tensor<?x?xf32>
  //      CHECK: return %[[masked_write]] : tensor<?x?xf32>
  %cst = arith.constant 42.43 : f32
  %c0 = arith.constant 0 : index
  %1 = tensor.pad %0 low[0, %c0] high[%h0, %h1]  {
    ^bb0(%hh1: index, %hh2: index):
      tensor.yield %cst : f32
    } : tensor<?x?xf32> to tensor<?x?xf32>
  return %1: tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [2, 4] : !transform.any_op
    transform.yield
  }
}

// -----
// This case is supported because low padding `%l0` is applied on
// a unit dimension which is supported, non unit result dimension low
// padding is currently unsupported.
//  CHECK-LABEL: func @test_masked_vectorize_non_zero_low_pad_unit_res_dim
func.func @test_masked_vectorize_non_zero_low_pad_unit_res_dim(
  %0 : tensor<?x?xf32>, %h0 : index, %h1 : index, %l0 : index)
    -> tensor<1x4xf32>
{
  //  CHECK-DAG: %[[C42:.*]] = arith.constant 4.243000e+01 : f32
  //  CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
  //      CHECK: %[[C0_1:.*]] = arith.constant 0 : index
  //  CHECK-DAG: %[[D0:.*]] = tensor.dim {{.*}} : tensor<?x?xf32>
  //  CHECK-DAG: %[[D1:.*]] = tensor.dim {{.*}} : tensor<?x?xf32>
  //      CHECK: %[[MASK:.*]] = vector.create_mask %[[D0]], %[[D1]] : vector<1x4xi1>
  //      CHECK: %[[MASKED_READ:.*]] = vector.mask %[[MASK]] {
  // CHECK-SAME:   vector.transfer_read %{{.*}}[%[[C0_1]], %[[C0_1]]], %[[C42]]
  // CHECK-SAME:   {in_bounds = [true, true]} : tensor<?x?xf32>, vector<1x4xf32>
  // CHECK-SAME: } : vector<1x4xi1> -> vector<1x4xf32>
  //  CHECK-DAG: %[[EMPTY:.*]] = tensor.empty() : tensor<1x4xf32>
  //  CHECK-DAG: %[[C0_2:.*]] = arith.constant 0 : index
  //      CHECK: %[[MASKED_WRITE:.*]] = vector.transfer_write %[[MASKED_READ]], %[[EMPTY]][%[[C0_2]], %[[C0_2]]]
  // CHECK-SAME:   {in_bounds = [true, true]} : vector<1x4xf32>, tensor<1x4xf32>
  //      CHECK: return %[[MASKED_WRITE]] : tensor<1x4xf32>
  %cst = arith.constant 42.43 : f32
  %c0 = arith.constant 0 : index
  %1 = tensor.pad %0 low[%l0, %c0] high[%h0, %h1]  {
    ^bb0(%hh1: index, %hh2: index):
      tensor.yield %cst : f32
    } : tensor<?x?xf32> to tensor<1x4xf32>
  return %1: tensor<1x4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.pad"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [1, 4] : !transform.any_op
    transform.yield
  }
}
