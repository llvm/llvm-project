// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

///----------------------------------------------------------------------------------------
/// Tests for `linalg::vectorize` with caller-provided mask bounds.
///
/// A mask bound forces masking of an iteration space dimension even when that
/// dimension is statically sized. This is for operands that were padded to a
/// static shape but whose trailing elements must not contribute to the result.
/// It is the dual of `assume_dynamic_dims_match_vec_sizes`,
/// which suppresses masking of a dynamic dim.
///----------------------------------------------------------------------------------------

/// A statically shaped reduction, masked on the reduction dim (1) with a
/// dynamic bound. Without the bound no mask would be created at all, since the
/// operand shape matches the vector shape exactly - see
/// @unmasked_static_reduction below.

// CHECK-LABEL: func.func @masked_static_reduction(
// CHECK-SAME:      %[[SRC:.*]]: tensor<128x64xf32>,
// CHECK-SAME:      %[[ACC:.*]]: tensor<128xf32>,
// CHECK-SAME:      %[[UB:.*]]: index
// CHECK:         %[[BOUND:.*]] = affine.min
// CHECK-DAG:     %[[C128:.*]] = arith.constant 128 : index
// CHECK:         %[[MASK:.*]] = vector.create_mask %[[C128]], %[[BOUND]] : vector<128x64xi1>
// CHECK:         %[[READ:.*]] = vector.mask %[[MASK]] {
// CHECK-SAME:      vector.transfer_read %[[SRC]]
// CHECK-SAME:    } : vector<128x64xi1> -> vector<128x64xf32>
// CHECK:         vector.mask %[[MASK]] {
// CHECK-SAME:      vector.multi_reduction <maximumf>, %[[READ]]
// CHECK-SAME:    } : vector<128x64xi1> -> vector<128xf32>
func.func @masked_static_reduction(%src: tensor<128x64xf32>,
                                   %acc: tensor<128xf32>,
                                   %ub: index) -> tensor<128xf32> {
  // The bound is <= 64, so masking is meaningful (and provably in range).
  %bound = affine.min affine_map<()[s0] -> (s0, 64)>()[%ub]
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%src : tensor<128x64xf32>) outs(%acc : tensor<128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.maximumf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %min = transform.structured.match ops{["affine.min"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %bound = transform.get_result %min[0]
      : (!transform.any_op) -> !transform.any_value
    transform.structured.vectorize %generic vector_sizes [128, 64]
      mask_bounds [1] (%bound : !transform.any_value) : !transform.any_op
    transform.yield
  }
}

// -----

/// Baseline: without a mask bound, a fully static reduction is vectorized
/// unmasked. This is the behavior a mask bound opts out of.

// CHECK-LABEL: func.func @unmasked_static_reduction(
// CHECK-NOT:     vector.create_mask
// CHECK-NOT:     vector.mask
// CHECK:         vector.multi_reduction <maximumf>
func.func @unmasked_static_reduction(%src: tensor<128x64xf32>,
                                     %acc: tensor<128xf32>) -> tensor<128xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%src : tensor<128x64xf32>) outs(%acc : tensor<128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.maximumf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %generic vector_sizes [128, 64] : !transform.any_op
    transform.yield
  }
}

// -----

/// A bound on the parallel dim (0) instead of the reduction dim. The bound
/// lands in the leading position of the mask, and the accumulator read/write -
/// which only maps dim 0 - is masked too.

// CHECK-LABEL: func.func @masked_parallel_dim(
// CHECK-SAME:      %[[SRC:.*]]: tensor<128x64xf32>,
// CHECK-SAME:      %[[ACC:.*]]: tensor<128xf32>,
// CHECK-SAME:      %[[UB:.*]]: index
// CHECK:         %[[BOUND:.*]] = affine.min
// CHECK-DAG:     %[[C64:.*]] = arith.constant 64 : index
// CHECK:         %[[MASK:.*]] = vector.create_mask %[[BOUND]], %[[C64]] : vector<128x64xi1>
// CHECK:         %[[ACC_MASK:.*]] = vector.create_mask %[[BOUND]] : vector<128xi1>
// CHECK:         vector.mask %[[MASK]] {
// CHECK-SAME:      vector.multi_reduction <add>
func.func @masked_parallel_dim(%src: tensor<128x64xf32>,
                               %acc: tensor<128xf32>,
                               %ub: index) -> tensor<128xf32> {
  %bound = affine.min affine_map<()[s0] -> (s0, 128)>()[%ub]
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%src : tensor<128x64xf32>) outs(%acc : tensor<128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %min = transform.structured.match ops{["affine.min"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %bound = transform.get_result %min[0]
      : (!transform.any_op) -> !transform.any_value
    transform.structured.vectorize %generic vector_sizes [128, 64]
      mask_bounds [0] (%bound : !transform.any_value) : !transform.any_op
    transform.yield
  }
}

// -----

/// Bounds on several dims at once, listed out of order to check that
/// `mask_bound_dims` indexes the iteration space rather than the operand order.

// CHECK-LABEL: func.func @masked_multiple_dims(
// CHECK:         %[[B0:.*]] = affine.min
// CHECK:         %[[B1:.*]] = affine.min
// CHECK:         %[[MASK:.*]] = vector.create_mask %[[B0]], %[[B1]] : vector<128x64xi1>
// CHECK:         vector.mask %[[MASK]] { vector.multi_reduction <add>
func.func @masked_multiple_dims(%src: tensor<128x64xf32>,
                                %acc: tensor<128xf32>,
                                %ub0: index, %ub1: index) -> tensor<128xf32> {
  %b0 = affine.min affine_map<()[s0] -> (s0, 128)>()[%ub0]
  %b1 = affine.min affine_map<()[s0] -> (s0, 64)>()[%ub1]
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%src : tensor<128x64xf32>) outs(%acc : tensor<128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %mins = transform.structured.match ops{["affine.min"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %min0, %min1 = transform.split_handle %mins
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %b0 = transform.get_result %min0[0]
      : (!transform.any_op) -> !transform.any_value
    %b1 = transform.get_result %min1[0]
      : (!transform.any_op) -> !transform.any_value
    transform.structured.vectorize %generic vector_sizes [128, 64]
      mask_bounds [1, 0] (%b1, %b0 : !transform.any_value, !transform.any_value)
      : !transform.any_op
    transform.yield
  }
}

// -----

/// An elementwise op (no reduction) with a bound: the bound reaches the
/// transfer_read and transfer_write, not just a reduction.

// CHECK-LABEL: func.func @masked_elementwise(
// CHECK:         %[[BOUND:.*]] = affine.min
// CHECK:         %[[MASK:.*]] = vector.create_mask %[[BOUND]] : vector<128xi1>
// CHECK:         vector.mask %[[MASK]] { vector.transfer_read
// CHECK:         vector.mask %[[MASK]] { vector.transfer_write
func.func @masked_elementwise(%src: tensor<128xf32>, %ub: index) -> tensor<128xf32> {
  %bound = affine.min affine_map<()[s0] -> (s0, 128)>()[%ub]
  %empty = tensor.empty() : tensor<128xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%src : tensor<128xf32>) outs(%empty : tensor<128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.negf %in : f32
    linalg.yield %1 : f32
  } -> tensor<128xf32>
  return %0 : tensor<128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %min = transform.structured.match ops{["affine.min"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %bound = transform.get_result %min[0]
      : (!transform.any_op) -> !transform.any_value
    transform.structured.vectorize %generic vector_sizes [128]
      mask_bounds [0] (%bound : !transform.any_value) : !transform.any_op
    transform.yield
  }
}

// -----

/// A bound on a dim that is *already* dynamic. The bound wins over the
/// `tensor.dim`-derived size that would otherwise be inferred.

// CHECK-LABEL: func.func @masked_bound_overrides_dynamic_dim(
// CHECK:         %[[BOUND:.*]] = affine.min
// CHECK:         %[[MASK:.*]] = vector.create_mask %[[BOUND]] : vector<128xi1>
// CHECK-NOT:     vector.create_mask
// CHECK:         vector.mask %[[MASK]]
func.func @masked_bound_overrides_dynamic_dim(%src: tensor<?xf32>,
                                              %ub: index) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %bound = affine.min affine_map<()[s0] -> (s0, 128)>()[%ub]
  %dim = tensor.dim %src, %c0 : tensor<?xf32>
  %empty = tensor.empty(%dim) : tensor<?xf32>
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } ins(%src : tensor<?xf32>) outs(%empty : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.negf %in : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %min = transform.structured.match ops{["affine.min"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %bound = transform.get_result %min[0]
      : (!transform.any_op) -> !transform.any_value
    transform.structured.vectorize %generic vector_sizes [128]
      mask_bounds [0] (%bound : !transform.any_value) : !transform.any_op
    transform.yield
  }
}

// -----

/// A bound takes precedence over `assume_dynamic_dims_match_vec_sizes`, which
/// would otherwise suppress masking entirely. Dim 1 carries an explicit bound
/// and so is still masked. The precedence is per masking map, not per op: the
/// accumulator only projects onto dim 0, which has no bound, so it keeps the
/// unmasked in-bounds form the assumption asks for.

// CHECK-LABEL: func.func @masked_bound_beats_assume_dynamic_dims(
// CHECK:         %[[BOUND:.*]] = affine.min
// CHECK:         %[[DIM:.*]] = tensor.dim
// CHECK:         %[[MASK:.*]] = vector.create_mask %[[DIM]], %[[BOUND]] : vector<128x64xi1>
// CHECK:         vector.mask %[[MASK]] { vector.transfer_read
// The accumulator read is left unmasked and in-bounds, not wrapped in a mask.
// CHECK:         vector.transfer_read {{.*}}in_bounds = [true]{{.*}} tensor<?xf32>
// CHECK-NOT:     vector.create_mask
// CHECK:         vector.mask %[[MASK]] { vector.multi_reduction <maximumf>
func.func @masked_bound_beats_assume_dynamic_dims(%src: tensor<?x64xf32>,
                                                  %acc: tensor<?xf32>,
                                                  %ub: index) -> tensor<?xf32> {
  %bound = affine.min affine_map<()[s0] -> (s0, 64)>()[%ub]
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%src : tensor<?x64xf32>) outs(%acc : tensor<?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.maximumf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %generic = transform.structured.match ops{["linalg.generic"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %min = transform.structured.match ops{["affine.min"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %bound = transform.get_result %min[0]
      : (!transform.any_op) -> !transform.any_value
    transform.structured.vectorize %generic vector_sizes [128, 64]
      mask_bounds [1] (%bound : !transform.any_value)
      {assume_dynamic_dims_match_vec_sizes} : !transform.any_op
    transform.yield
  }
}

// -----

/// A fixed-shape op (`linalg.matmul`) whose result
/// feeds two reductions with *different* neutral values. `maximumf` needs -inf,
/// `minimumf` needs +inf. Padding the operands up to the instruction's fixed shape
/// satisfies the shape requirement, but no single padding value satisfies both
/// neutral values, so the reductions have to be masked to the number of valid
/// columns instead.

// CHECK-LABEL: func.func @masked_reductions_after_matmul(
/// The valid extent, captured before padding.
// CHECK:         %[[DIM:.*]] = tensor.dim
// CHECK:         %[[BOUND:.*]] = affine.min
// CHECK:         %[[PADDED:.*]] = tensor.pad
/// The matmul is statically shaped, so it is vectorized without any masking.
// CHECK:         vector.transfer_read %[[PADDED]]
// CHECK:         %[[MUL:.*]] = arith.mulf
// CHECK:         %[[MM:.*]] = vector.multi_reduction <add>, %[[MUL]]
// CHECK:         %[[MM_RES:.*]] = vector.transfer_write %[[MM]]
// CHECK:         %[[MASK_MAX:.*]] = vector.create_mask %{{.*}}, %[[BOUND]] : vector<128x64xi1>
// CHECK:         %[[READ_MAX:.*]] = vector.mask %[[MASK_MAX]] {
// CHECK-SAME:      vector.transfer_read %[[MM_RES]]
// CHECK-SAME:    } : vector<128x64xi1> -> vector<128x64xf32>
// CHECK:         vector.mask %[[MASK_MAX]] {
// CHECK-SAME:      vector.multi_reduction <maximumf>, %[[READ_MAX]]
// CHECK-SAME:    } : vector<128x64xi1> -> vector<128xf32>
// CHECK:         %[[MASK_MIN:.*]] = vector.create_mask %{{.*}}, %[[BOUND]] : vector<128x64xi1>
// CHECK:         %[[READ_MIN:.*]] = vector.mask %[[MASK_MIN]] {
// CHECK-SAME:      vector.transfer_read %[[MM_RES]]
// CHECK-SAME:    } : vector<128x64xi1> -> vector<128x64xf32>
// CHECK:         vector.mask %[[MASK_MIN]] {
// CHECK-SAME:      vector.multi_reduction <minimumf>, %[[READ_MIN]]
// CHECK-SAME:    } : vector<128x64xi1> -> vector<128xf32>
func.func @masked_reductions_after_matmul(%A: tensor<128x32xf32>,
                                          %B_dyn: tensor<32x?xf32>,
                                          %C: tensor<128x64xf32>,
                                          %max_init: tensor<128xf32>,
                                          %sum_init: tensor<128xf32>)
    -> (tensor<128xf32>, tensor<128xf32>) {
  %c1 = arith.constant 1 : index
  %pad_val = arith.constant 0.0 : f32
  %dim = tensor.dim %B_dyn, %c1 : tensor<32x?xf32>
  %extent = affine.min affine_map<()[s0] -> (s0, 64)>()[%dim]
  %high = affine.apply affine_map<()[s0] -> (64 - s0)>()[%extent]
  %B = tensor.pad %B_dyn low[0, 0] high[0, %high] {
  ^bb0(%i: index, %j: index):
    tensor.yield %pad_val : f32
  } : tensor<32x?xf32> to tensor<32x64xf32>
  %mm = linalg.matmul ins(%A, %B : tensor<128x32xf32>, tensor<32x64xf32>)
                      outs(%C : tensor<128x64xf32>) -> tensor<128x64xf32>
  %max = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%mm : tensor<128x64xf32>) outs(%max_init : tensor<128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %m = arith.maximumf %in, %out : f32
    linalg.yield %m : f32
  } -> tensor<128xf32>
  %sum = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                     affine_map<(d0, d1) -> (d0)>],
    iterator_types = ["parallel", "reduction"]
  } ins(%mm : tensor<128x64xf32>) outs(%sum_init : tensor<128xf32>) {
  ^bb0(%in: f32, %out: f32):
    %s = arith.minimumf %in, %out : f32
    linalg.yield %s : f32
  } -> tensor<128xf32>
  return %max, %sum : tensor<128xf32>, tensor<128xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %reductions = transform.structured.match ops{["linalg.generic"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %min = transform.structured.match ops{["affine.min"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    %extent = transform.get_result %min[0]
      : (!transform.any_op) -> !transform.any_value
    // The fixed-shape op needs no bound: its shapes are static and complete.
    transform.structured.vectorize %matmul vector_sizes [128, 64, 32]
      : !transform.any_op
    // The reductions must only cover the valid columns.
    transform.structured.vectorize %reductions vector_sizes [128, 64]
      mask_bounds [1] (%extent : !transform.any_value) : !transform.any_op
    transform.yield
  }
}
