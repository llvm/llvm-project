// RUN: mlir-opt %s -linalg-fuse-elementwise-ops -split-input-file | FileCheck %s
// RUN: mlir-opt %s -convert-elementwise-to-linalg -linalg-fuse-elementwise-ops -split-input-file | FileCheck %s --check-prefix=PIPELINE

#identity = affine_map<(d0, d1) -> (d0, d1)>
#transpose = affine_map<(d0, d1) -> (d1, d0)>

// CHECK-LABEL: func.func @split_elementwise_with_concat
// CHECK-SAME: %[[A0:.+]]: tensor<2x3xf32>, %[[A1:.+]]: tensor<2x4xf32>,
// CHECK-SAME: %[[B0:.+]]: tensor<3x2xf32>, %[[B1:.+]]: tensor<4x2xf32>,
// CHECK-SAME: %[[S:.+]]: f32
// CHECK-NOT: tensor.concat
// CHECK: %[[EMPTY0:.+]] = tensor.empty() : tensor<2x3xf32>
// CHECK: %[[PART0:.+]] = linalg.generic
// CHECK-SAME: ins(%[[A0]], %[[B0]], %[[S]] : tensor<2x3xf32>, tensor<3x2xf32>, f32)
// CHECK-SAME: outs(%[[EMPTY0]] : tensor<2x3xf32>)
// CHECK: %[[EMPTY1:.+]] = tensor.empty() : tensor<2x4xf32>
// CHECK: %[[PART1:.+]] = linalg.generic
// CHECK-SAME: ins(%[[A1]], %[[B1]], %[[S]] : tensor<2x4xf32>, tensor<4x2xf32>, f32)
// CHECK-SAME: outs(%[[EMPTY1]] : tensor<2x4xf32>)
// CHECK: %[[RESULT:.+]] = tensor.concat dim(1) %[[PART0]], %[[PART1]]
// CHECK-SAME: (tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x7xf32>
// CHECK: return %[[RESULT]]
func.func @split_elementwise_with_concat(
    %a0: tensor<2x3xf32>, %a1: tensor<2x4xf32>,
    %b0: tensor<3x2xf32>, %b1: tensor<4x2xf32>, %s: f32)
    -> tensor<2x7xf32> {
  %a = tensor.concat dim(1) %a0, %a1
      : (tensor<2x3xf32>, tensor<2x4xf32>) -> tensor<2x7xf32>
  %b = tensor.concat dim(0) %b0, %b1
      : (tensor<3x2xf32>, tensor<4x2xf32>) -> tensor<7x2xf32>
  %empty = tensor.empty() : tensor<2x7xf32>
  %result = linalg.generic {
      indexing_maps = [#identity, #transpose,
                       affine_map<(d0, d1) -> ()>, #identity],
      iterator_types = ["parallel", "parallel"]}
      ins(%a, %b, %s : tensor<2x7xf32>, tensor<7x2xf32>, f32)
      outs(%empty : tensor<2x7xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %scalar: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      %scaled = arith.mulf %sum, %scalar : f32
      linalg.yield %scaled : f32
  } -> tensor<2x7xf32>
  return %result : tensor<2x7xf32>
}

// -----

#identity = affine_map<(d0) -> (d0)>

// Preserve a destination that is read by slicing it at the same boundaries.
// CHECK-LABEL: func.func @split_elementwise_with_accumulator
// CHECK: %[[INIT0:.+]] = tensor.extract_slice %[[INIT:.+]][0] [3] [1]
// CHECK: %[[PART0:.+]] = linalg.generic
// CHECK-SAME: outs(%[[INIT0]] : tensor<3xf32>)
// CHECK: %[[INIT1:.+]] = tensor.extract_slice %[[INIT]][3] [4] [1]
// CHECK: %[[PART1:.+]] = linalg.generic
// CHECK-SAME: outs(%[[INIT1]] : tensor<4xf32>)
// CHECK: tensor.concat dim(0) %[[PART0]], %[[PART1]]
func.func @split_elementwise_with_accumulator(
    %a0: tensor<3xf32>, %a1: tensor<4xf32>, %init: tensor<7xf32>)
    -> tensor<7xf32> {
  %a = tensor.concat dim(0) %a0, %a1
      : (tensor<3xf32>, tensor<4xf32>) -> tensor<7xf32>
  %result = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel"]}
      ins(%a : tensor<7xf32>) outs(%init : tensor<7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
  } -> tensor<7xf32>
  return %result : tensor<7xf32>
}

// -----

#identity = affine_map<(d0) -> (d0)>

// Splitting the consumer makes its elementwise producers directly fusable.
// CHECK-LABEL: func.func @split_exposes_elementwise_fusion
// CHECK-COUNT-2: linalg.generic
// CHECK-NOT: linalg.generic
// CHECK: tensor.concat dim(0)
func.func @split_exposes_elementwise_fusion(
    %x0: tensor<3xf32>, %x1: tensor<4xf32>,
    %y0: tensor<3xf32>, %y1: tensor<4xf32>) -> tensor<7xf32> {
  %c = arith.constant 2.0 : f32
  %x0e = tensor.empty() : tensor<3xf32>
  %px0 = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel"]}
      ins(%x0 : tensor<3xf32>) outs(%x0e : tensor<3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.mulf %in, %c : f32
      linalg.yield %v : f32
  } -> tensor<3xf32>
  %x1e = tensor.empty() : tensor<4xf32>
  %px1 = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel"]}
      ins(%x1 : tensor<4xf32>) outs(%x1e : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.mulf %in, %c : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  %y0e = tensor.empty() : tensor<3xf32>
  %py0 = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel"]}
      ins(%y0 : tensor<3xf32>) outs(%y0e : tensor<3xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.mulf %in, %c : f32
      linalg.yield %v : f32
  } -> tensor<3xf32>
  %y1e = tensor.empty() : tensor<4xf32>
  %py1 = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel"]}
      ins(%y1 : tensor<4xf32>) outs(%y1e : tensor<4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %v = arith.mulf %in, %c : f32
      linalg.yield %v : f32
  } -> tensor<4xf32>
  %x = tensor.concat dim(0) %px0, %px1
      : (tensor<3xf32>, tensor<4xf32>) -> tensor<7xf32>
  %y = tensor.concat dim(0) %py0, %py1
      : (tensor<3xf32>, tensor<4xf32>) -> tensor<7xf32>
  %empty = tensor.empty() : tensor<7xf32>
  %result = linalg.generic {
      indexing_maps = [#identity, #identity, #identity],
      iterator_types = ["parallel"]}
      ins(%x, %y : tensor<7xf32>, tensor<7xf32>)
      outs(%empty : tensor<7xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
  } -> tensor<7xf32>
  return %result : tensor<7xf32>
}

// -----

#identity = affine_map<(d0) -> (d0)>

// Dynamic partition boundaries cannot be proven to line up.
// CHECK-LABEL: func.func @dynamic_partitions_not_split
// CHECK-COUNT-2: tensor.concat
// CHECK: linalg.generic
func.func @dynamic_partitions_not_split(
    %a0: tensor<?xf32>, %a1: tensor<?xf32>,
    %b0: tensor<?xf32>, %b1: tensor<?xf32>) -> tensor<?xf32> {
  %a = tensor.concat dim(0) %a0, %a1
      : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %b = tensor.concat dim(0) %b0, %b1
      : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %c0 = arith.constant 0 : index
  %dim = tensor.dim %a, %c0 : tensor<?xf32>
  %empty = tensor.empty(%dim) : tensor<?xf32>
  %result = linalg.generic {
      indexing_maps = [#identity, #identity, #identity],
      iterator_types = ["parallel"]}
      ins(%a, %b : tensor<?xf32>, tensor<?xf32>)
      outs(%empty : tensor<?xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}

// -----

#identity = affine_map<(d0) -> (d0)>

// Reductions are not elementwise and reordering them could change numerical
// behavior.
// CHECK-LABEL: func.func @reduction_not_split
// CHECK: tensor.concat
// CHECK: linalg.generic
func.func @reduction_not_split(%a0: tensor<3xf32>, %a1: tensor<4xf32>)
    -> tensor<f32> {
  %a = tensor.concat dim(0) %a0, %a1
      : (tensor<3xf32>, tensor<4xf32>) -> tensor<7xf32>
  %empty = tensor.empty() : tensor<f32>
  %result = linalg.generic {
      indexing_maps = [#identity, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]}
      ins(%a : tensor<7xf32>) outs(%empty : tensor<f32>) {
    ^bb0(%in: f32, %out: f32):
      %sum = arith.addf %in, %out : f32
      linalg.yield %sum : f32
  } -> tensor<f32>
  return %result : tensor<f32>
}

// -----

#identity = affine_map<(d0) -> (d0)>

// Splitting would reset linalg.index at every partition boundary.
// CHECK-LABEL: func.func @index_semantics_not_split
// CHECK: tensor.concat
// CHECK: linalg.generic
// CHECK: linalg.index
func.func @index_semantics_not_split(%a0: tensor<3xindex>,
                                     %a1: tensor<4xindex>) -> tensor<7xindex> {
  %a = tensor.concat dim(0) %a0, %a1
      : (tensor<3xindex>, tensor<4xindex>) -> tensor<7xindex>
  %empty = tensor.empty() : tensor<7xindex>
  %result = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel"]}
      ins(%a : tensor<7xindex>) outs(%empty : tensor<7xindex>) {
    ^bb0(%in: index, %out: index):
      %index = linalg.index 0 : index
      %sum = arith.addi %in, %index : index
      linalg.yield %sum : index
  } -> tensor<7xindex>
  return %result : tensor<7xindex>
}

// -----

#identity = affine_map<(d0) -> (d0)>

// Different concat counts require slicing one input at a boundary introduced
// by the other concat, which this pattern intentionally does not do.
// CHECK-LABEL: func.func @different_partition_counts_not_split
// CHECK-COUNT-2: tensor.concat
// CHECK: linalg.generic
func.func @different_partition_counts_not_split(
    %a0: tensor<3xf32>, %a1: tensor<4xf32>,
    %b0: tensor<3xf32>, %b1: tensor<2xf32>, %b2: tensor<2xf32>)
    -> tensor<7xf32> {
  %a = tensor.concat dim(0) %a0, %a1
      : (tensor<3xf32>, tensor<4xf32>) -> tensor<7xf32>
  %b = tensor.concat dim(0) %b0, %b1, %b2
      : (tensor<3xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<7xf32>
  %empty = tensor.empty() : tensor<7xf32>
  %result = linalg.generic {
      indexing_maps = [#identity, #identity, #identity],
      iterator_types = ["parallel"]}
      ins(%a, %b : tensor<7xf32>, tensor<7xf32>)
      outs(%empty : tensor<7xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
  } -> tensor<7xf32>
  return %result : tensor<7xf32>
}

// -----

#identity = affine_map<(d0) -> (d0)>

// Equal concat counts are insufficient when their boundaries differ.
// CHECK-LABEL: func.func @different_partition_sizes_not_split
// CHECK-COUNT-2: tensor.concat
// CHECK: linalg.generic
func.func @different_partition_sizes_not_split(
    %a0: tensor<3xf32>, %a1: tensor<4xf32>,
    %b0: tensor<2xf32>, %b1: tensor<5xf32>) -> tensor<7xf32> {
  %a = tensor.concat dim(0) %a0, %a1
      : (tensor<3xf32>, tensor<4xf32>) -> tensor<7xf32>
  %b = tensor.concat dim(0) %b0, %b1
      : (tensor<2xf32>, tensor<5xf32>) -> tensor<7xf32>
  %empty = tensor.empty() : tensor<7xf32>
  %result = linalg.generic {
      indexing_maps = [#identity, #identity, #identity],
      iterator_types = ["parallel"]}
      ins(%a, %b : tensor<7xf32>, tensor<7xf32>)
      outs(%empty : tensor<7xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
  } -> tensor<7xf32>
  return %result : tensor<7xf32>
}

// -----

#identity = affine_map<(d0) -> (d0)>

// Keep a concat with another live consumer instead of duplicating the
// elementwise computation while retaining the original concat.
// CHECK-LABEL: func.func @concat_with_another_consumer_not_split
// CHECK: tensor.concat
// CHECK: call @consume_tensor
// CHECK: linalg.generic
func.func private @consume_tensor(%arg: tensor<7xf32>)

func.func @concat_with_another_consumer_not_split(
    %a0: tensor<3xf32>, %a1: tensor<4xf32>) -> tensor<7xf32> {
  %a = tensor.concat dim(0) %a0, %a1
      : (tensor<3xf32>, tensor<4xf32>) -> tensor<7xf32>
  func.call @consume_tensor(%a) : (tensor<7xf32>) -> ()
  %empty = tensor.empty() : tensor<7xf32>
  %result = linalg.generic {
      indexing_maps = [#identity, #identity],
      iterator_types = ["parallel"]}
      ins(%a : tensor<7xf32>) outs(%empty : tensor<7xf32>) {
    ^bb0(%in: f32, %out: f32):
      %twice = arith.addf %in, %in : f32
      linalg.yield %twice : f32
  } -> tensor<7xf32>
  return %result : tensor<7xf32>
}

// -----

#identity = affine_map<(d0, d1) -> (d0, d1)>
#broadcast_second_dim = affine_map<(d0, d1) -> (d1)>

// A non-concat tensor that is broadcast along the split dimension can be
// reused by every partition.
// CHECK-LABEL: func.func @broadcast_input_is_reused
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[A0:.+]], %[[B:.+]] : tensor<3x2xf32>, tensor<2xf32>)
// CHECK: linalg.generic
// CHECK-SAME: ins(%[[A1:.+]], %[[B]] : tensor<4x2xf32>, tensor<2xf32>)
// CHECK: tensor.concat dim(0)
func.func @broadcast_input_is_reused(
    %a0: tensor<3x2xf32>, %a1: tensor<4x2xf32>, %b: tensor<2xf32>)
    -> tensor<7x2xf32> {
  %a = tensor.concat dim(0) %a0, %a1
      : (tensor<3x2xf32>, tensor<4x2xf32>) -> tensor<7x2xf32>
  %empty = tensor.empty() : tensor<7x2xf32>
  %result = linalg.generic {
      indexing_maps = [#identity, #broadcast_second_dim, #identity],
      iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : tensor<7x2xf32>, tensor<2xf32>)
      outs(%empty : tensor<7x2xf32>) {
    ^bb0(%in: f32, %broadcast: f32, %out: f32):
      %sum = arith.addf %in, %broadcast : f32
      linalg.yield %sum : f32
  } -> tensor<7x2xf32>
  return %result : tensor<7x2xf32>
}

// -----

#identity = affine_map<(d0, d1) -> (d0, d1)>

// A non-concat tensor that varies along the split dimension would need slices.
// CHECK-LABEL: func.func @varying_non_concat_input_not_split
// CHECK: tensor.concat
// CHECK: linalg.generic
func.func @varying_non_concat_input_not_split(
    %a0: tensor<3x2xf32>, %a1: tensor<4x2xf32>, %b: tensor<7x2xf32>)
    -> tensor<7x2xf32> {
  %a = tensor.concat dim(0) %a0, %a1
      : (tensor<3x2xf32>, tensor<4x2xf32>) -> tensor<7x2xf32>
  %empty = tensor.empty() : tensor<7x2xf32>
  %result = linalg.generic {
      indexing_maps = [#identity, #identity, #identity],
      iterator_types = ["parallel", "parallel"]}
      ins(%a, %b : tensor<7x2xf32>, tensor<7x2xf32>)
      outs(%empty : tensor<7x2xf32>) {
    ^bb0(%lhs: f32, %rhs: f32, %out: f32):
      %sum = arith.addf %lhs, %rhs : f32
      linalg.yield %sum : f32
  } -> tensor<7x2xf32>
  return %result : tensor<7x2xf32>
}

// -----

#identity = affine_map<(d0, d1) -> (d0, d1)>
#transpose = affine_map<(d0, d1) -> (d1, d0)>

// The output concat dimension comes from the output map, not directly from
// the split loop dimension: splitting d0 maps to output dim(1) here.
// CHECK-LABEL: func.func @transpose_output_concat_dimension
// CHECK: linalg.generic
// CHECK-SAME: outs({{.*}} : tensor<2x3xf32>)
// CHECK: linalg.generic
// CHECK-SAME: outs({{.*}} : tensor<2x4xf32>)
// CHECK: tensor.concat dim(1)
func.func @transpose_output_concat_dimension(
    %a0: tensor<3x2xf32>, %a1: tensor<4x2xf32>) -> tensor<2x7xf32> {
  %a = tensor.concat dim(0) %a0, %a1
      : (tensor<3x2xf32>, tensor<4x2xf32>) -> tensor<7x2xf32>
  %empty = tensor.empty() : tensor<2x7xf32>
  %result = linalg.generic {
      indexing_maps = [#identity, #transpose],
      iterator_types = ["parallel", "parallel"]}
      ins(%a : tensor<7x2xf32>) outs(%empty : tensor<2x7xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  } -> tensor<2x7xf32>
  return %result : tensor<2x7xf32>
}

// -----

// The source-level form from the issue is converted to linalg before this
// rewrite runs.
// PIPELINE-LABEL: func.func @source_level_elementwise
// PIPELINE: linalg.generic
// PIPELINE: linalg.generic
// PIPELINE: tensor.concat dim(0)
func.func @source_level_elementwise(
    %x0: tensor<3xf32>, %x1: tensor<4xf32>,
    %y0: tensor<3xf32>, %y1: tensor<4xf32>) -> tensor<7xf32> {
  %x = tensor.concat dim(0) %x0, %x1
      : (tensor<3xf32>, tensor<4xf32>) -> tensor<7xf32>
  %y = tensor.concat dim(0) %y0, %y1
      : (tensor<3xf32>, tensor<4xf32>) -> tensor<7xf32>
  %result = arith.addf %x, %y : tensor<7xf32>
  return %result : tensor<7xf32>
}
