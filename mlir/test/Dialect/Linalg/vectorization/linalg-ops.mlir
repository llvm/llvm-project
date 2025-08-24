// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

///----------------------------------------------------------------------------------------
/// Tests for linalg.generic
///----------------------------------------------------------------------------------------

func.func @vectorize_dynamic_identity(%arg0: tensor<?xf32>,
                                      %arg1: tensor<?xf32>,
                                      %arg2: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic { indexing_maps = [affine_map<(d0) -> (d0)>,
                                         affine_map<(d0) -> (d0)>,
                                         affine_map<(d0) -> (d0)>],
                   iterator_types = ["parallel"] }
    ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
    outs(%arg2 : tensor<?xf32>) {
    ^bb(%in0: f32, %in1: f32, %out: f32) :
      %0 = arith.addf %in0, %in1 : f32
      linalg.yield %0 : f32
    } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL:   @vectorize_dynamic_identity
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = tensor.dim %{{.*}}, %[[VAL_3]] : tensor<?xf32>
// CHECK:           %[[VAL_7:.*]] = vector.create_mask %[[VAL_4]] : vector<4xi1>
// CHECK:           %[[VAL_8:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %{{.*}} {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
// CHECK:           %[[VAL_10:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %{{.*}} {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
// CHECK:           %[[VAL_12:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %{{.*}} {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
// CHECK:           %[[VAL_13:.*]] = arith.addf %[[VAL_8]], %[[VAL_10]] : vector<4xf32>
// CHECK:           %[[VAL_14:.*]] = vector.mask %[[VAL_7]] { vector.transfer_write %{{.*}} {in_bounds = [true]} : vector<4xf32>, tensor<?xf32> } : vector<4xi1> -> tensor<?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [4] : !transform.any_op
    transform.yield
  }
}

// -----

func.func @vectorize_dynamic_identity_scalable(%arg0: tensor<?xf32>,
                                               %arg1: tensor<?xf32>,
                                               %arg2: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic { indexing_maps = [affine_map<(d0) -> (d0)>,
                                         affine_map<(d0) -> (d0)>,
                                         affine_map<(d0) -> (d0)>],
                   iterator_types = ["parallel"] }
    ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
    outs(%arg2 : tensor<?xf32>) {
    ^bb(%in0: f32, %in1: f32, %out: f32) :
      %0 = arith.addf %in0, %in1 : f32
      linalg.yield %0 : f32
    } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL:   @vectorize_dynamic_identity_scalable
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = tensor.dim %{{.*}}, %[[VAL_3]] : tensor<?xf32>
// CHECK:           %[[VAL_7:.*]] = vector.create_mask %[[VAL_4]] : vector<[4]xi1>
// CHECK:           %[[VAL_8:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %{{.*}} {in_bounds = [true]} : tensor<?xf32>, vector<[4]xf32> } : vector<[4]xi1> -> vector<[4]xf32>
// CHECK:           %[[VAL_10:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %{{.*}} {in_bounds = [true]} : tensor<?xf32>, vector<[4]xf32> } : vector<[4]xi1> -> vector<[4]xf32>
// CHECK:           %[[VAL_12:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %{{.*}} {in_bounds = [true]} : tensor<?xf32>, vector<[4]xf32> } : vector<[4]xi1> -> vector<[4]xf32>
// CHECK:           %[[VAL_13:.*]] = arith.addf %[[VAL_8]], %[[VAL_10]] : vector<[4]xf32>
// CHECK:           %[[VAL_14:.*]] = vector.mask %[[VAL_7]] { vector.transfer_write %{{.*}} {in_bounds = [true]} : vector<[4]xf32>, tensor<?xf32> } : vector<[4]xi1> -> tensor<?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [[4]] : !transform.any_op
    transform.yield
  }
}

// -----

func.func @vectorize_dynamic_identity_with_constant(%arg0: tensor<?xf32>,
                                                    %arg1: tensor<?xf32>,
                                                    %arg2: tensor<?xf32>) -> tensor<?xf32> {
  %c4 = arith.constant 4 : index
  %0 = linalg.generic { indexing_maps = [affine_map<(d0) -> (d0)>,
                                         affine_map<(d0) -> (d0)>,
                                         affine_map<(d0) -> (d0)>],
                   iterator_types = ["parallel"] }
    ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
    outs(%arg2 : tensor<?xf32>) {
    ^bb(%in0: f32, %in1: f32, %out: f32) :
      %0 = arith.addf %in0, %in1 : f32
      linalg.yield %0 : f32
    } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL:   @vectorize_dynamic_identity_with_constant
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = tensor.dim %{{.*}}, %[[VAL_3]] : tensor<?xf32>
// CHECK:           %[[VAL_7:.*]] = vector.create_mask %[[VAL_4]] : vector<4xi1>
// CHECK:           %[[VAL_8:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %{{.*}} {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
// CHECK:           %[[VAL_10:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %{{.*}} {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
// CHECK:           %[[VAL_12:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %{{.*}} {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
// CHECK:           %[[VAL_13:.*]] = arith.addf %[[VAL_8]], %[[VAL_10]] : vector<4xf32>
// CHECK:           %[[VAL_14:.*]] = vector.mask %[[VAL_7]] { vector.transfer_write %{{.*}} {in_bounds = [true]} : vector<4xf32>, tensor<?xf32> } : vector<4xi1> -> tensor<?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %size = transform.structured.match ops{["arith.constant"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [%size] : !transform.any_op, !transform.any_op
    transform.yield
  }
}

// -----

func.func @vectorize_dynamic_identity_with_param(%arg0: tensor<?xf32>,
                                                 %arg1: tensor<?xf32>,
                                                 %arg2: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic { indexing_maps = [affine_map<(d0) -> (d0)>,
                                         affine_map<(d0) -> (d0)>,
                                         affine_map<(d0) -> (d0)>],
                   iterator_types = ["parallel"] }
    ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
    outs(%arg2 : tensor<?xf32>) {
    ^bb(%in0: f32, %in1: f32, %out: f32) :
      %0 = arith.addf %in0, %in1 : f32
      linalg.yield %0 : f32
    } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL:   @vectorize_dynamic_identity_with_param
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = tensor.dim %{{.*}}, %[[VAL_3]] : tensor<?xf32>
// CHECK:           %[[VAL_7:.*]] = vector.create_mask %[[VAL_4]] : vector<4xi1>
// CHECK:           %[[VAL_8:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %{{.*}} {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
// CHECK:           %[[VAL_10:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %{{.*}} {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
// CHECK:           %[[VAL_12:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %{{.*}} {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
// CHECK:           %[[VAL_13:.*]] = arith.addf %[[VAL_8]], %[[VAL_10]] : vector<4xf32>
// CHECK:           %[[VAL_14:.*]] = vector.mask %[[VAL_7]] { vector.transfer_write %{{.*}} {in_bounds = [true]} : vector<4xf32>, tensor<?xf32> } : vector<4xi1> -> tensor<?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %vector_size = transform.param.constant 4 : i64 -> !transform.param<i64>
    transform.structured.vectorize %0 vector_sizes [%vector_size] : !transform.any_op, !transform.param<i64>
    transform.yield
  }
}

// -----

func.func @vectorize_dynamic_1d_broadcast(%arg0: tensor<?xf32>,
                                          %arg1: tensor<?xf32>,
                                          %arg2: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic { indexing_maps = [affine_map<(d0) -> (0)>,
                                         affine_map<(d0) -> (d0)>,
                                         affine_map<(d0) -> (d0)>],
                        iterator_types = ["parallel"] }
    ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
    outs(%arg2 : tensor<?xf32>) {
    ^bb(%in0: f32, %in1: f32, %out: f32) :
      %0 = arith.addf %in0, %in1 : f32
      linalg.yield %0 : f32
    } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL:   @vectorize_dynamic_1d_broadcast
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = tensor.dim %{{.*}}, %[[VAL_3]] : tensor<?xf32>
// CHECK:           %[[VAL_7:.*]] = vector.transfer_read %{{.*}} {permutation_map = #{{.*}}} : tensor<?xf32>, vector<4xf32>
// CHECK:           %[[VAL_9:.*]] = vector.create_mask %[[VAL_4]] : vector<4xi1>
// CHECK:           %[[VAL_10:.*]] = vector.mask %[[VAL_9]] { vector.transfer_read %{{.*}} {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
// CHECK:           %[[VAL_12:.*]] = vector.mask %[[VAL_9]] { vector.transfer_read %{{.*}} {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
// CHECK:           %[[VAL_13:.*]] = arith.addf %[[VAL_7]], %[[VAL_10]] : vector<4xf32>
// CHECK:           %[[VAL_14:.*]] = vector.mask %{{.*}} { vector.transfer_write %[[VAL_13]], {{.*}} {in_bounds = [true]} : vector<4xf32>, tensor<?xf32> } : vector<4xi1> -> tensor<?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [4] : !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0, 0)>

func.func @dynamic_generic_with_reduction_and_broadcast(%arg0: tensor<?x?xf32>, %init: tensor<?x?xf32>) -> (tensor<?x?xf32>) {
  %0 = linalg.generic { indexing_maps = [#map, #map1],
                        iterator_types = ["parallel", "reduction"]}
    ins(%arg0 : tensor<?x?xf32>)
    outs(%init : tensor<?x?xf32>) {
  ^bb0(%in: f32, %out: f32):
    %1 = arith.addf %in, %out : f32
    linalg.yield %1 : f32
  } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL:   func.func @dynamic_generic_with_reduction_and_broadcast(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:      %[[VAL_1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = tensor.dim %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf32>
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = ub.poison : f32
// CHECK:           %[[VAL_8:.*]] = vector.create_mask %[[VAL_3]], %[[VAL_5]] : vector<4x4xi1>
// CHECK:           %[[VAL_9:.*]] = vector.mask %[[VAL_8]] { vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_6]], %[[VAL_6]]], %[[VAL_7]] {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x4xf32> } : vector<4x4xi1> -> vector<4x4xf32>
// CHECK:           %[[VAL_10:.*]] = ub.poison : f32
// CHECK:           %[[VAL_11:.*]] = vector.create_mask %[[VAL_3]] : vector<4xi1>
// CHECK:           %[[VAL_12:.*]] = vector.mask %[[VAL_11]] { vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_6]], %[[VAL_6]]], %[[VAL_10]] {in_bounds = [true], permutation_map = #[[$MAP]]} : tensor<?x?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
// CHECK:           %[[VAL_13:.*]] = vector.mask %[[VAL_8]] { vector.multi_reduction <add>, %[[VAL_9]], %[[VAL_12]] [1] : vector<4x4xf32> to vector<4xf32> } : vector<4x4xi1> -> vector<4xf32>
// CHECK:           %[[VAL_14:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_15:.*]] = vector.mask %[[VAL_11]] { vector.transfer_write %[[VAL_13]], %[[VAL_1]]{{\[}}%[[VAL_14]], %[[VAL_14]]] {in_bounds = [true], permutation_map = #[[$MAP]]} : vector<4xf32>, tensor<?x?xf32> } : vector<4xi1> -> tensor<?x?xf32>
// CHECK:           return %[[VAL_15]] : tensor<?x?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [4, 4] : !transform.any_op
    transform.yield
  }
}

// -----

func.func @vectorize_dynamic_2d_transpose(%arg0: tensor<?x?xf32>,
                                          %arg1: tensor<?x?xf32>,
                                          %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic { indexing_maps = [affine_map<(d0, d1) -> (d1, d0)>,
                                         affine_map<(d0, d1) -> (d0, d1)>,
                                         affine_map<(d0, d1) -> (d0, d1)>],
                        iterator_types = ["parallel", "parallel"] }
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2 : tensor<?x?xf32>) {
    ^bb(%in0: f32, %in1: f32, %out: f32) :
      %0 = arith.addf %in0, %in1 : f32
      linalg.yield %0 : f32
    } -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
}

// CHECK-LABEL:   @vectorize_dynamic_2d_transpose
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_4:.*]] = tensor.dim %{{.*}}, %[[VAL_3]] : tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_6:.*]] = tensor.dim %{{.*}}, %[[VAL_5]] : tensor<?x?xf32>
// CHECK:           %[[VAL_9:.*]] = vector.create_mask %[[VAL_6]], %[[VAL_4]] : vector<8x4xi1>
// CHECK:           %[[VAL_10:.*]] = vector.mask %[[VAL_9]] { vector.transfer_read %{{.*}} {in_bounds = [true, true], permutation_map = #{{.*}}} : tensor<?x?xf32>, vector<4x8xf32> } : vector<8x4xi1> -> vector<4x8xf32>
// CHECK:           %[[VAL_12:.*]] = vector.create_mask %[[VAL_4]], %[[VAL_6]] : vector<4x8xi1>
// CHECK:           %[[VAL_13:.*]] = vector.mask %[[VAL_12]] { vector.transfer_read %{{.*}} {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x8xf32> } : vector<4x8xi1> -> vector<4x8xf32>
// CHECK:           %[[VAL_14:.*]] = ub.poison : f32
// CHECK:           %[[VAL_15:.*]] = vector.mask %[[VAL_12]] { vector.transfer_read %{{.*}} {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x8xf32> } : vector<4x8xi1> -> vector<4x8xf32>
// CHECK:           %[[VAL_16:.*]] = arith.addf %[[VAL_10]], %[[VAL_13]] : vector<4x8xf32>
// CHECK:           %[[VAL_17:.*]] = vector.mask %[[VAL_12]] { vector.transfer_write %[[VAL_16]], %{{.*}} {in_bounds = [true, true]} : vector<4x8xf32>, tensor<?x?xf32> } : vector<4x8xi1> -> tensor<?x?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [4, 8] : !transform.any_op
    transform.yield
  }
}

// -----

func.func @vectorize_dynamic_generic_2d_broadcast(%arg0: tensor<?x?xf32>,
                                                  %arg1: tensor<?x?xf32>,
                                                  %arg2: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic { indexing_maps = [affine_map<(d0, d1) -> (0, d1)>,
                                         affine_map<(d0, d1) -> (d0, d1)>,
                                         affine_map<(d0, d1) -> (d0, d1)>],
                        iterator_types = ["parallel", "parallel"] }
    ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
    outs(%arg2 : tensor<?x?xf32>) {
    ^bb(%in0: f32, %in1: f32, %out: f32) :
      %0 = arith.addf %in0, %in1 : f32
      linalg.yield %0 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL:   @vectorize_dynamic_generic_2d_broadcast
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = tensor.dim %{{.*}}, %[[VAL_3]] : tensor<?x?xf32>
// CHECK:           %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_6:.*]] = tensor.dim %{{.*}}, %[[VAL_5]] : tensor<?x?xf32>
// CHECK:           %[[VAL_9:.*]] = vector.create_mask %[[VAL_6]] : vector<8xi1>
// CHECK:           %[[VAL_10:.*]] = vector.mask %[[VAL_9]] { vector.transfer_read %{{.*}} {in_bounds = [true, true], permutation_map = #{{.*}}} : tensor<?x?xf32>, vector<4x8xf32> } : vector<8xi1> -> vector<4x8xf32>
// CHECK:           %[[VAL_12:.*]] = vector.create_mask %[[VAL_4]], %[[VAL_6]] : vector<4x8xi1>
// CHECK:           %[[VAL_13:.*]] = vector.mask %[[VAL_12]] { vector.transfer_read %{{.*}} {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x8xf32> } : vector<4x8xi1> -> vector<4x8xf32>
// CHECK:           %[[VAL_15:.*]] = vector.mask %[[VAL_12]] { vector.transfer_read %{{.*}} {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x8xf32> } : vector<4x8xi1> -> vector<4x8xf32>
// CHECK:           %[[VAL_16:.*]] = arith.addf %[[VAL_10]], %[[VAL_13]] : vector<4x8xf32>
// CHECK:           %[[VAL_18:.*]] = vector.mask %[[VAL_12]] { vector.transfer_write %{{.*}} {in_bounds = [true, true]} : vector<4x8xf32>, tensor<?x?xf32> } : vector<4x8xi1> -> tensor<?x?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [4, 8] : !transform.any_op
    transform.yield
  }
}

// -----

func.func @vectorize_dynamic_reduction_2d(%arg0: tensor<?x?xf32>,
                                          %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic { indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                         affine_map<(d0, d1) -> (d0)>],
                        iterator_types = ["parallel", "reduction"] }
    ins(%arg0 : tensor<?x?xf32>)
    outs(%arg1 : tensor<?xf32>) {
    ^bb(%in: f32, %out: f32) :
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
    } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [4, 8] : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL:   @vectorize_dynamic_reduction_2d(
// CHECK-SAME:                                 %[[VAL_0:.*]]: tensor<?x?xf32>,
// CHECK-SAME:                                 %[[VAL_1:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?xf32>
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = tensor.dim %[[VAL_0]], %[[VAL_4]] : tensor<?x?xf32>
// CHECK:           %[[VAL_8:.*]] = vector.create_mask %[[VAL_3]], %[[VAL_5]] : vector<4x8xi1>
// CHECK:           %[[VAL_9:.*]] = vector.mask %[[VAL_8]] { vector.transfer_read %[[VAL_0]]{{.*}} {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x8xf32> } : vector<4x8xi1> -> vector<4x8xf32>
// CHECK:           %[[VAL_11:.*]] = vector.create_mask %[[VAL_3]] : vector<4xi1>
// CHECK:           %[[VAL_12:.*]] = vector.mask %[[VAL_11]] { vector.transfer_read %[[VAL_1]]{{.*}} {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
// CHECK:           %[[VAL_13:.*]] = vector.mask %[[VAL_8]] { vector.multi_reduction <add>, %[[VAL_9]], %[[VAL_12]] [1] : vector<4x8xf32> to vector<4xf32> } : vector<4x8xi1> -> vector<4xf32>
// CHECK:           %[[VAL_15:.*]] = vector.mask %[[VAL_11]] { vector.transfer_write %[[VAL_13]], %[[VAL_1]]{{.*}} {in_bounds = [true]} : vector<4xf32>, tensor<?xf32> } : vector<4xi1> -> tensor<?xf32>
// CHECK:           return %[[VAL_15]] : tensor<?xf32>
// CHECK:         }

// -----

func.func @vectorize_dynamic_reduction_2d_scalable(%arg0: tensor<?x?xf32>,
                                                   %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic { indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                         affine_map<(d0, d1) -> (d0)>],
                        iterator_types = ["parallel", "reduction"] }
    ins(%arg0 : tensor<?x?xf32>)
    outs(%arg1 : tensor<?xf32>) {
    ^bb(%in: f32, %out: f32) :
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
    } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL:  func.func @vectorize_dynamic_reduction_2d_scalable(
// CHECK-SAME:     %[[ARG_0:.*]]: tensor<?x?xf32>, %[[ARG_1:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:    %[[C0_IDX:.*]] = arith.constant 0 : index
// CHECK:    %[[DIM_A0_0:.*]] = tensor.dim %[[ARG_0]], %[[C0_IDX]] : tensor<?x?xf32>
// CHECK:    %[[C1_IDX:.*]] = arith.constant 1 : index
// CHECK:    %[[DIM_A0_1:.*]] = tensor.dim %[[ARG_0]], %[[C1_IDX]] : tensor<?x?xf32>
// CHECK:    %[[C0_IDX:.*]] = arith.constant 0 : index
// CHECK:    %[[PV:.*]] = ub.poison : f32
// CHECK:    %[[MASK_2D:.*]] = vector.create_mask %[[DIM_A0_0]], %[[DIM_A0_1]] : vector<4x[8]xi1>
// CHECK:    %[[VEC_RD_0:.*]] = vector.mask %[[MASK_2D]] { vector.transfer_read %[[ARG_0]][%[[C0_IDX]], %[[C0_IDX]]], %[[PV]] {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x[8]xf32> } : vector<4x[8]xi1> -> vector<4x[8]xf32>
// CHECK:    %[[PV:.*]] = ub.poison : f32
// CHECK:    %[[MASK_1D:.*]] = vector.create_mask %[[DIM_A0_0]] : vector<4xi1>
// CHECK:    %[[VEC_RD_1:.*]] = vector.mask %[[MASK_1D]] { vector.transfer_read %[[ARG_1]][%[[C0_IDX]]], %[[PV]] {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
// CHECK:    %[[REDUCE:.*]] = vector.mask %[[MASK_2D]] { vector.multi_reduction <add>, %[[VEC_RD_0]], %[[VEC_RD_1]] [1] : vector<4x[8]xf32> to vector<4xf32> } : vector<4x[8]xi1> -> vector<4xf32>
// CHECK:    %[[C0_IDX:.*]] = arith.constant 0 : index
// CHECK:    %{{.*}} = vector.mask %[[MASK_1D]] { vector.transfer_write %[[REDUCE]], %[[ARG_1]][%[[C0_IDX]]] {in_bounds = [true]} : vector<4xf32>, tensor<?xf32> } : vector<4xi1> -> tensor<?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [4, [8]] : !transform.any_op
    transform.yield
  }
}

// -----

func.func @vectorize_dynamic_reduction_scalable_1d(%arg0: tensor<?xf32>,
                                                   %arg1: tensor<f32>) -> tensor<f32> {

  %0 = linalg.reduce ins(%arg0 : tensor<?xf32>) outs(%arg1 : tensor<f32>) dimensions = [0]
  (%in: f32, %init: f32) {
    %0 = arith.addf %in, %init : f32
    linalg.yield %0 : f32
  }
  return %0 : tensor<f32>
}

// CHECK-LABEL:  func.func @vectorize_dynamic_reduction_scalable_1d(
// CHECK-SAME:     %[[ARG_0:.*]]: tensor<?xf32>, %[[ARG_1:.*]]: tensor<f32>) -> tensor<f32> {
// CHECK:          %[[C0_IDX:.*]] = arith.constant 0 : index
// CHECK:          %[[DIM_A0_0:.*]] = tensor.dim %[[ARG_0]], %[[C0_IDX]] : tensor<?xf32>
// CHECK:          %[[C0_IDX:.*]] = arith.constant 0 : index
// CHECK:          %[[PV:.*]] = ub.poison : f32
// CHECK:          %[[MASK:.*]] = vector.create_mask %[[DIM_A0_0]] : vector<[4]xi1>
// CHECK:          %[[VEC_RD_0:.*]] = vector.mask %[[MASK]] { vector.transfer_read %[[ARG_0]][%[[C0_IDX]]], %[[PV]] {in_bounds = [true]} : tensor<?xf32>, vector<[4]xf32> } : vector<[4]xi1> -> vector<[4]xf32>
// CHECK:          %[[PV:.*]] = ub.poison : f32
// CHECK:          %[[VEC_RD_1:.*]] = vector.transfer_read %[[ARG_1]][], %[[PV]] : tensor<f32>, vector<f32>
// CHECK:          %[[ACC_f32:.*]] = vector.extract %[[VEC_RD_1]][] : f32 from vector<f32>
// CHECK:          %[[REDUCE:.*]] = vector.mask %[[MASK]] { vector.multi_reduction <add>, %[[VEC_RD_0]], %[[ACC_f32]] [0] : vector<[4]xf32> to f32 } : vector<[4]xi1> -> f32
// CHECK:          %[[VEC_f32:.*]] = vector.broadcast %[[REDUCE]] : f32 to vector<f32>
// CHECK:          %{{.*}} = vector.transfer_write %[[VEC_f32]], %[[ARG_1]][] : vector<f32>, tensor<f32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.reduce"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [[4]] : !transform.any_op
    transform.yield
  }
}

// -----

func.func @vectorize_dynamic_transpose_reduction(%arg0: tensor<?x?x?xf32>,
                                                 %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic { indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                                         affine_map<(d0, d1, d2) -> (d2, d1)>],
                        iterator_types = ["reduction", "parallel", "parallel"] }
    ins(%arg0 : tensor<?x?x?xf32>)
    outs(%arg1 : tensor<?x?xf32>) {
    ^bb(%in: f32, %out: f32) :
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [4, 8, 16] : !transform.any_op
    transform.yield
  }
}

// CHECK-LABEL:   @vectorize_dynamic_transpose_reduction(
// CHECK-SAME:                                           %[[VAL_0:.*]]: tensor<?x?x?xf32>,
// CHECK-SAME:                                           %[[VAL_1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = tensor.dim %[[VAL_0]], %[[VAL_4]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_6:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_7:.*]] = tensor.dim %[[VAL_0]], %[[VAL_6]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_10:.*]] = vector.create_mask %[[VAL_3]], %[[VAL_5]], %[[VAL_7]] : vector<4x8x16xi1>
// CHECK:           %[[VAL_11:.*]] = vector.mask %[[VAL_10]] { vector.transfer_read %[[VAL_0]]{{.*}} {in_bounds = [true, true, true]} : tensor<?x?x?xf32>, vector<4x8x16xf32> } : vector<4x8x16xi1> -> vector<4x8x16xf32>
// CHECK:           %[[VAL_13:.*]] = vector.create_mask %[[VAL_7]], %[[VAL_5]] : vector<16x8xi1>
// CHECK:           %[[VAL_14:.*]] = vector.mask %[[VAL_13]] { vector.transfer_read %[[VAL_1]]{{.*}} {in_bounds = [true, true], permutation_map = #{{.*}}} : tensor<?x?xf32>, vector<8x16xf32> } : vector<16x8xi1> -> vector<8x16xf32>
// CHECK:           %[[VAL_15:.*]] = vector.mask %[[VAL_10]] { vector.multi_reduction <add>, %[[VAL_11]], %[[VAL_14]] [0] : vector<4x8x16xf32> to vector<8x16xf32> } : vector<4x8x16xi1> -> vector<8x16xf32>
// CHECK:           %[[VAL_17:.*]] = vector.mask %[[VAL_13]] { vector.transfer_write %[[VAL_15]], %{{.*}} {in_bounds = [true, true], permutation_map = #{{.*}}} : vector<8x16xf32>, tensor<?x?xf32> } : vector<16x8xi1> -> tensor<?x?xf32>

// -----

func.func @vectorize_dynamic_transpose_reduction_with_params(%arg0: tensor<?x?x?xf32>,
                                                             %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.generic { indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                                         affine_map<(d0, d1, d2) -> (d2, d1)>],
                        iterator_types = ["reduction", "parallel", "parallel"] }
    ins(%arg0 : tensor<?x?x?xf32>)
    outs(%arg1 : tensor<?x?xf32>) {
    ^bb(%in: f32, %out: f32) :
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
    } -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %vector_size_0 = transform.param.constant 4 : i64 -> !transform.param<i64>
    %vector_size_2 = transform.param.constant 16 : i64 -> !transform.param<i64>
    transform.structured.vectorize %0 vector_sizes
      [%vector_size_0, 8, %vector_size_2] : !transform.any_op, !transform.param<i64>, !transform.param<i64>
    transform.yield
  }
}

// CHECK-LABEL:   @vectorize_dynamic_transpose_reduction_with_params(
// CHECK-SAME:                                           %[[VAL_0:.*]]: tensor<?x?x?xf32>,
// CHECK-SAME:                                           %[[VAL_1:.*]]: tensor<?x?xf32>) -> tensor<?x?xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_5:.*]] = tensor.dim %[[VAL_0]], %[[VAL_4]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_6:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_7:.*]] = tensor.dim %[[VAL_0]], %[[VAL_6]] : tensor<?x?x?xf32>
// CHECK:           %[[VAL_10:.*]] = vector.create_mask %[[VAL_3]], %[[VAL_5]], %[[VAL_7]] : vector<4x8x16xi1>
// CHECK:           %[[VAL_11:.*]] = vector.mask %[[VAL_10]] { vector.transfer_read %[[VAL_0]]{{.*}} {in_bounds = [true, true, true]} : tensor<?x?x?xf32>, vector<4x8x16xf32> } : vector<4x8x16xi1> -> vector<4x8x16xf32>
// CHECK:           %[[VAL_13:.*]] = vector.create_mask %[[VAL_7]], %[[VAL_5]] : vector<16x8xi1>
// CHECK:           %[[VAL_14:.*]] = vector.mask %[[VAL_13]] { vector.transfer_read %[[VAL_1]]{{.*}} {in_bounds = [true, true], permutation_map = #{{.*}}} : tensor<?x?xf32>, vector<8x16xf32> } : vector<16x8xi1> -> vector<8x16xf32>
// CHECK:           %[[VAL_15:.*]] = vector.mask %[[VAL_10]] { vector.multi_reduction <add>, %[[VAL_11]], %[[VAL_14]] [0] : vector<4x8x16xf32> to vector<8x16xf32> } : vector<4x8x16xi1> -> vector<8x16xf32>
// CHECK:           %[[VAL_17:.*]] = vector.mask %[[VAL_13]] { vector.transfer_write %[[VAL_15]], %{{.*}} {in_bounds = [true, true], permutation_map = #{{.*}}} : vector<8x16xf32>, tensor<?x?xf32> } : vector<16x8xi1> -> tensor<?x?xf32>

// -----

func.func @vectorize_partial_dynamic_identity(%arg0: tensor<8x?xf32>,
                                              %arg1: tensor<8x?xf32>,
                                              %arg2: tensor<8x?xf32>) -> tensor<8x?xf32> {
  %0 = linalg.generic { indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                         affine_map<(d0, d1) -> (d0, d1)>,
                                         affine_map<(d0, d1) -> (d0, d1)>],
                   iterator_types = ["parallel", "parallel"] }
    ins(%arg0, %arg1 : tensor<8x?xf32>, tensor<8x?xf32>)
    outs(%arg2 : tensor<8x?xf32>) {
    ^bb(%in0: f32, %in1: f32, %out: f32) :
      %0 = arith.addf %in0, %in1 : f32
      linalg.yield %0 : f32
    } -> tensor<8x?xf32>
  return %0 : tensor<8x?xf32>
}

// CHECK-LABEL:   func.func @vectorize_partial_dynamic_identity(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<8x?xf32>, %[[VAL_1:.*]]: tensor<8x?xf32>, %[[VAL_2:.*]]: tensor<8x?xf32>) -> tensor<8x?xf32> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<8x?xf32>
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_6:.*]] = ub.poison : f32
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_8:.*]] = vector.create_mask %[[VAL_7]], %[[VAL_4]] : vector<8x32xi1>
// CHECK:           %[[VAL_9:.*]] = vector.mask %[[VAL_8]] { vector.transfer_read %[[VAL_0]][%[[VAL_5]], %[[VAL_5]]], %[[VAL_6]] {in_bounds = [true, true]} : tensor<8x?xf32>, vector<8x32xf32> } : vector<8x32xi1> -> vector<8x32xf32>
// CHECK:           %[[VAL_10:.*]] = ub.poison : f32
// CHECK:           %[[VAL_11:.*]] = vector.mask %[[VAL_8]] { vector.transfer_read %[[VAL_1]][%[[VAL_5]], %[[VAL_5]]], %[[VAL_10]] {in_bounds = [true, true]} : tensor<8x?xf32>, vector<8x32xf32> } : vector<8x32xi1> -> vector<8x32xf32>
// CHECK:           %[[VAL_12:.*]] = ub.poison : f32
// CHECK:           %[[VAL_13:.*]] = vector.mask %[[VAL_8]] { vector.transfer_read %[[VAL_2]][%[[VAL_5]], %[[VAL_5]]], %[[VAL_12]] {in_bounds = [true, true]} : tensor<8x?xf32>, vector<8x32xf32> } : vector<8x32xi1> -> vector<8x32xf32>
// CHECK:           %[[VAL_14:.*]] = arith.addf %[[VAL_9]], %[[VAL_11]] : vector<8x32xf32>
// CHECK:           %[[VAL_15:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_16:.*]] = vector.mask %[[VAL_8]] { vector.transfer_write %[[VAL_14]], %[[VAL_2]][%[[VAL_15]], %[[VAL_15]]] {in_bounds = [true, true]} : vector<8x32xf32>, tensor<8x?xf32> } : vector<8x32xi1> -> tensor<8x?xf32>


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [8, 32] : !transform.any_op
    transform.yield
  }
}

// -----

func.func @vectorize_partial_dynamic_identity_scalable(%arg0: tensor<8x?xf32>,
                                                       %arg1: tensor<8x?xf32>,
                                                       %arg2: tensor<8x?xf32>) -> tensor<8x?xf32> {
  %0 = linalg.generic { indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                         affine_map<(d0, d1) -> (d0, d1)>,
                                         affine_map<(d0, d1) -> (d0, d1)>],
                   iterator_types = ["parallel", "parallel"] }
    ins(%arg0, %arg1 : tensor<8x?xf32>, tensor<8x?xf32>)
    outs(%arg2 : tensor<8x?xf32>) {
    ^bb(%in0: f32, %in1: f32, %out: f32) :
      %0 = arith.addf %in0, %in1 : f32
      linalg.yield %0 : f32
    } -> tensor<8x?xf32>
  return %0 : tensor<8x?xf32>
}

// CHECK-LABEL:   func.func @vectorize_partial_dynamic_identity_scalable
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<8x?xf32>, %[[VAL_1:.*]]: tensor<8x?xf32>, %[[VAL_2:.*]]: tensor<8x?xf32>) -> tensor<8x?xf32> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = tensor.dim %[[VAL_0]], %[[VAL_3]] : tensor<8x?xf32>
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_6:.*]] = ub.poison : f32
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_8:.*]] = vector.create_mask %[[VAL_7]], %[[VAL_4]] : vector<8x[32]xi1>
// CHECK:           %[[VAL_9:.*]] = vector.mask %[[VAL_8]] { vector.transfer_read %[[VAL_0]][%[[VAL_5]], %[[VAL_5]]], %[[VAL_6]] {in_bounds = [true, true]} : tensor<8x?xf32>, vector<8x[32]xf32> } : vector<8x[32]xi1> -> vector<8x[32]xf32>
// CHECK:           %[[VAL_10:.*]] = ub.poison : f32
// CHECK:           %[[VAL_11:.*]] = vector.mask %[[VAL_8]] { vector.transfer_read %[[VAL_1]][%[[VAL_5]], %[[VAL_5]]], %[[VAL_10]] {in_bounds = [true, true]} : tensor<8x?xf32>, vector<8x[32]xf32> } : vector<8x[32]xi1> -> vector<8x[32]xf32>
// CHECK:           %[[VAL_12:.*]] = ub.poison : f32
// CHECK:           %[[VAL_13:.*]] = vector.mask %[[VAL_8]] { vector.transfer_read %[[VAL_2]][%[[VAL_5]], %[[VAL_5]]], %[[VAL_12]] {in_bounds = [true, true]} : tensor<8x?xf32>, vector<8x[32]xf32> } : vector<8x[32]xi1> -> vector<8x[32]xf32>
// CHECK:           %[[VAL_14:.*]] = arith.addf %[[VAL_9]], %[[VAL_11]] : vector<8x[32]xf32>
// CHECK:           %[[VAL_15:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_16:.*]] = vector.mask %[[VAL_8]] { vector.transfer_write %[[VAL_14]], %[[VAL_2]][%[[VAL_15]], %[[VAL_15]]] {in_bounds = [true, true]} : vector<8x[32]xf32>, tensor<8x?xf32> } : vector<8x[32]xi1> -> tensor<8x?xf32>


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [8, [32]] : !transform.any_op
    transform.yield
  }
}

// -----

func.func @do_not_generate_masks(%arg0: tensor<8x32xf32>,
                                 %arg1: tensor<8x32xf32>,
                                 %arg2: tensor<8x32xf32>) -> tensor<8x32xf32> {
  %0 = linalg.generic { indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                         affine_map<(d0, d1) -> (d0, d1)>,
                                         affine_map<(d0, d1) -> (d0, d1)>],
                   iterator_types = ["parallel", "parallel"] }
    ins(%arg0, %arg1 : tensor<8x32xf32>, tensor<8x32xf32>)
    outs(%arg2 : tensor<8x32xf32>) {
    ^bb(%in0: f32, %in1: f32, %out: f32) :
      %0 = arith.addf %in0, %in1 : f32
      linalg.yield %0 : f32
    } -> tensor<8x32xf32>
  return %0 : tensor<8x32xf32>
}

// CHECK-LABEL: func.func @do_not_generate_masks
// CHECK-NOT: vector.mask

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [8, 32] : !transform.any_op
    transform.yield
  }
}

// -----

func.func @vectorize_static_shape_with_mask(%arg0: tensor<8x30xf32>,
                                            %arg1: tensor<8x30xf32>,
                                            %arg2: tensor<8x30xf32>) -> tensor<8x30xf32> {
  %0 = linalg.generic { indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                         affine_map<(d0, d1) -> (d0, d1)>,
                                         affine_map<(d0, d1) -> (d0, d1)>],
                   iterator_types = ["parallel", "parallel"] }
    ins(%arg0, %arg1 : tensor<8x30xf32>, tensor<8x30xf32>)
    outs(%arg2 : tensor<8x30xf32>) {
    ^bb(%in0: f32, %in1: f32, %out: f32) :
      %0 = arith.addf %in0, %in1 : f32
      linalg.yield %0 : f32
    } -> tensor<8x30xf32>
  return %0 : tensor<8x30xf32>
}

// CHECK-LABEL:   func.func @vectorize_static_shape_with_mask(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<8x30xf32>, %[[VAL_1:.*]]: tensor<8x30xf32>, %[[VAL_2:.*]]: tensor<8x30xf32>) -> tensor<8x30xf32> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = ub.poison : f32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 30 : index
// CHECK:           %[[VAL_7:.*]] = vector.create_mask %[[VAL_5]], %[[VAL_6]] : vector<8x32xi1>
// CHECK:           %[[VAL_8:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %[[VAL_0]][%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : tensor<8x30xf32>, vector<8x32xf32> } : vector<8x32xi1> -> vector<8x32xf32>
// CHECK:           %[[VAL_9:.*]] = ub.poison : f32
// CHECK:           %[[VAL_10:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %[[VAL_1]][%[[VAL_3]], %[[VAL_3]]], %[[VAL_9]] {in_bounds = [true, true]} : tensor<8x30xf32>, vector<8x32xf32> } : vector<8x32xi1> -> vector<8x32xf32>
// CHECK:           %[[VAL_11:.*]] = ub.poison : f32
// CHECK:           %[[VAL_12:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %[[VAL_2]][%[[VAL_3]], %[[VAL_3]]], %[[VAL_11]] {in_bounds = [true, true]} : tensor<8x30xf32>, vector<8x32xf32> } : vector<8x32xi1> -> vector<8x32xf32>
// CHECK:           %[[VAL_13:.*]] = arith.addf %[[VAL_8]], %[[VAL_10]] : vector<8x32xf32>
// CHECK:           %[[VAL_14:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_15:.*]] = vector.mask %[[VAL_7]] { vector.transfer_write %[[VAL_13]], %[[VAL_2]][%[[VAL_14]], %[[VAL_14]]] {in_bounds = [true, true]} : vector<8x32xf32>, tensor<8x30xf32> } : vector<8x32xi1> -> tensor<8x30xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [8, 32] : !transform.any_op
    transform.yield
  }
}

// -----

func.func @vectorize_static_shape_with_mask_scalable(%arg0: tensor<8x30xf32>,
                                                     %arg1: tensor<8x30xf32>,
                                                     %arg2: tensor<8x30xf32>) -> tensor<8x30xf32> {
  %0 = linalg.generic { indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                         affine_map<(d0, d1) -> (d0, d1)>,
                                         affine_map<(d0, d1) -> (d0, d1)>],
                   iterator_types = ["parallel", "parallel"] }
    ins(%arg0, %arg1 : tensor<8x30xf32>, tensor<8x30xf32>)
    outs(%arg2 : tensor<8x30xf32>) {
    ^bb(%in0: f32, %in1: f32, %out: f32) :
      %0 = arith.addf %in0, %in1 : f32
      linalg.yield %0 : f32
    } -> tensor<8x30xf32>
  return %0 : tensor<8x30xf32>
}

// CHECK-LABEL:   func.func @vectorize_static_shape_with_mask_scalable(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<8x30xf32>, %[[VAL_1:.*]]: tensor<8x30xf32>, %[[VAL_2:.*]]: tensor<8x30xf32>) -> tensor<8x30xf32> {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = ub.poison : f32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 30 : index
// CHECK:           %[[VAL_7:.*]] = vector.create_mask %[[VAL_5]], %[[VAL_6]] : vector<8x[32]xi1>
// CHECK:           %[[VAL_8:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %[[VAL_0]][%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : tensor<8x30xf32>, vector<8x[32]xf32> } : vector<8x[32]xi1> -> vector<8x[32]xf32>
// CHECK:           %[[VAL_9:.*]] = ub.poison : f32
// CHECK:           %[[VAL_10:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %[[VAL_1]][%[[VAL_3]], %[[VAL_3]]], %[[VAL_9]] {in_bounds = [true, true]} : tensor<8x30xf32>, vector<8x[32]xf32> } : vector<8x[32]xi1> -> vector<8x[32]xf32>
// CHECK:           %[[VAL_11:.*]] = ub.poison : f32
// CHECK:           %[[VAL_12:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %[[VAL_2]][%[[VAL_3]], %[[VAL_3]]], %[[VAL_11]] {in_bounds = [true, true]} : tensor<8x30xf32>, vector<8x[32]xf32> } : vector<8x[32]xi1> -> vector<8x[32]xf32>
// CHECK:           %[[VAL_13:.*]] = arith.addf %[[VAL_8]], %[[VAL_10]] : vector<8x[32]xf32>
// CHECK:           %[[VAL_14:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_15:.*]] = vector.mask %[[VAL_7]] { vector.transfer_write %[[VAL_13]], %[[VAL_2]][%[[VAL_14]], %[[VAL_14]]] {in_bounds = [true, true]} : vector<8x[32]xf32>, tensor<8x30xf32> } : vector<8x[32]xi1> -> tensor<8x30xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [8, [32]] : !transform.any_op
    transform.yield
  }
}

// -----

///----------------------------------------------------------------------------------------
/// Tests for linalg.matvec
///----------------------------------------------------------------------------------------

// Scalable _reduction_ dimension.

func.func @vectorize_dynamic_matvec_trailing_reduction_dim(%arg0: tensor<?x?xf32>,
                                                           %arg1: tensor<?xf32>,
                                                           %arg2: tensor<?xf32>) {
  linalg.matvec ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>)
                 outs(%arg2 : tensor<?xf32>) -> tensor<?xf32>
  return
}

// CHECK-LABEL:  func.func @vectorize_dynamic_matvec_trailing_reduction_dim(
// CHECK-SAME:     %[[ARG_0:.*]]: tensor<?x?xf32>, %[[ARG_1:.*]]: tensor<?xf32>, %[[ARG_2:.*]]: tensor<?xf32>) {
// CHECK:    %[[C0_idx:.*]] = arith.constant 0 : index
// CHECK:    %[[DIM_A0_0:.*]] = tensor.dim %[[ARG_0]], %[[C0_idx]] : tensor<?x?xf32>
// CHECK:    %[[C1_idx:.*]] = arith.constant 1 : index
// CHECK:    %[[DIM_A0_1:.*]] = tensor.dim %[[ARG_0]], %[[C1_idx]] : tensor<?x?xf32>
// CHECK:    %[[C0_idx:.*]] = arith.constant 0 : index
// CHECK:    %[[PV:.*]] = ub.poison : f32
// CHECK:    %[[MASK_2d:.*]] = vector.create_mask %[[DIM_A0_0]], %[[DIM_A0_1]] : vector<4x[4]xi1>
// CHECK:    %[[VEC_RD_0:.*]] = vector.mask %[[MASK_2d]] { vector.transfer_read %[[ARG_0]][%[[C0_idx]], %[[C0_idx]]], %[[PV]] {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x[4]xf32> } : vector<4x[4]xi1> -> vector<4x[4]xf32>
// CHECK:    %[[PV:.*]] = ub.poison : f32
// CHECK:    %[[MASK_d1:.*]] = vector.create_mask %[[DIM_A0_1]] : vector<[4]xi1>
// CHECK:    %[[VEC_RD_1:.*]] = vector.mask %[[MASK_d1]] { vector.transfer_read %[[ARG_1]][%[[C0_idx]]], %[[PV]] {in_bounds = [true, true], permutation_map = #map} : tensor<?xf32>, vector<4x[4]xf32> } : vector<[4]xi1> -> vector<4x[4]xf32>
// CHECK:    %[[PV:.*]] = ub.poison : f32
// CHECK:    %[[MASK_d2:.*]] = vector.create_mask %[[DIM_A0_0]] : vector<4xi1>
// CHECK:    %[[VEC_RD_2:.*]] = vector.mask %[[MASK_d2]] { vector.transfer_read %[[ARG_2]][%[[C0_idx]]], %[[PV]] {in_bounds = [true]} : tensor<?xf32>, vector<4xf32> } : vector<4xi1> -> vector<4xf32>
// CHECK:    %[[MUL:.*]] = arith.mulf %[[VEC_RD_0:.*]], %[[VEC_RD_1:.*]] : vector<4x[4]xf32>
// CHECK:    %[[REDUCE:.*]] = vector.mask %[[MASK_2d]] { vector.multi_reduction <add>, %[[MUL]], %[[VEC_RD_2]] [1] : vector<4x[4]xf32> to vector<4xf32> } : vector<4x[4]xi1> -> vector<4xf32>
// CHECK:    %[[C0_idx:.*]] = arith.constant 0 : index
// CHECK:    %{{.*}} = vector.mask %[[MASK_d2]] { vector.transfer_write %[[REDUCE]], %[[ARG_2]][%[[C0_idx]]] {in_bounds = [true]} : vector<4xf32>, tensor<?xf32> } : vector<4xi1> -> tensor<?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matvec"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [4, [4]] : !transform.any_op
    transform.yield
  }
}

// -----

// Scalable _parallel_ dimension.

func.func @vectorize_dynamic_matvec_trailing_reduction_dim(%arg0: tensor<?x?xf32>,
                                                           %arg1: tensor<?xf32>,
                                                           %arg2:
                                                           tensor<?xf32>) ->
                                                           tensor<?xf32>{
  %0 = linalg.matvec ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?xf32>)
                 outs(%arg2 : tensor<?xf32>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

// CHECK-LABEL:  func.func @vectorize_dynamic_matvec_trailing_reduction_dim(
// CHECK-SAME:     %[[ARG_0:.*]]: tensor<?x?xf32>, %[[ARG_1:.*]]: tensor<?xf32>, %[[ARG_2:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:    %[[C0_idx:.*]] = arith.constant 0 : index
// CHECK:    %[[DIM_A0_0:.*]] = tensor.dim %[[ARG_0]], %[[C0_idx]] : tensor<?x?xf32>
// CHECK:    %[[C1_idx:.*]] = arith.constant 1 : index
// CHECK:    %[[DIM_A0_1:.*]] = tensor.dim %[[ARG_0]], %[[C1_idx]] : tensor<?x?xf32>
// CHECK:    %[[C0_idx:.*]] = arith.constant 0 : index
// CHECK:    %[[PV:.*]] = ub.poison : f32
// CHECK:    %[[MASK_2d:.*]] = vector.create_mask %[[DIM_A0_0]], %[[DIM_A0_1]] : vector<[4]x4xi1>
// CHECK:    %[[VEC_RD_0:.*]] = vector.mask %[[MASK_2d]] { vector.transfer_read %[[ARG_0]][%[[C0_idx]], %[[C0_idx]]], %[[PV]] {in_bounds = [true, true]} : tensor<?x?xf32>, vector<[4]x4xf32> } : vector<[4]x4xi1> -> vector<[4]x4xf32>
// CHECK:    %[[PV:.*]] = ub.poison : f32
// CHECK:    %[[MASK_d1:.*]] = vector.create_mask %[[DIM_A0_1]] : vector<4xi1>
// CHECK:    %[[VEC_RD_1:.*]] = vector.mask %[[MASK_d1]] { vector.transfer_read %[[ARG_1]][%[[C0_idx]]], %[[PV]] {in_bounds = [true, true], permutation_map = #map} : tensor<?xf32>, vector<[4]x4xf32> } : vector<4xi1> -> vector<[4]x4xf32>
// CHECK:    %[[PV:.*]] = ub.poison : f32
// CHECK:    %[[MASK_d2:.*]] = vector.create_mask %[[DIM_A0_0]] : vector<[4]xi1>
// CHECK:    %[[VEC_RD_2:.*]] = vector.mask %[[MASK_d2]] { vector.transfer_read %[[ARG_2]][%[[C0_idx]]], %[[PV]] {in_bounds = [true]} : tensor<?xf32>, vector<[4]xf32> } : vector<[4]xi1> -> vector<[4]xf32>
// CHECK:    %[[MUL:.*]] = arith.mulf %[[VEC_RD_0:.*]], %[[VEC_RD_1:.*]] : vector<[4]x4xf32>
// CHECK:    %[[REDUCE:.*]] = vector.mask %[[MASK_2d]] { vector.multi_reduction <add>, %[[MUL]], %[[VEC_RD_2]] [1] : vector<[4]x4xf32> to vector<[4]xf32> } : vector<[4]x4xi1> -> vector<[4]xf32>
// CHECK:    %[[C0_idx:.*]] = arith.constant 0 : index
// CHECK:    %{{.*}} = vector.mask %[[MASK_d2]] { vector.transfer_write %[[REDUCE]], %[[ARG_2]][%[[C0_idx]]] {in_bounds = [true]} : vector<[4]xf32>, tensor<?xf32> } : vector<[4]xi1> -> tensor<?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matvec"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [[4], 4] : !transform.any_op
    transform.yield
  }
}

// -----

///----------------------------------------------------------------------------------------
/// Tests for linalg.index
///----------------------------------------------------------------------------------------

#map = affine_map<(d0) -> (d0)>
func.func @vectorize_linalg_index_scalable(%dest: tensor<?xindex>) -> tensor<?xindex> {
  %0 = linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel"]
  } outs(%dest : tensor<?xindex>) {
  ^bb0(%in: index):
    %1 = linalg.index 0 : index
    linalg.yield %1: index
  } -> tensor<?xindex>
  return %0 : tensor<?xindex>
}

// CHECK-LABEL:   func.func @vectorize_linalg_index_scalable(
// CHECK-SAME:      %[[DEST:.*]]: tensor<?xindex>) -> tensor<?xindex> {
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[D0:.*]] = tensor.dim %[[DEST]], %[[C0]] : tensor<?xindex>
// CHECK:           %[[C0_1:.*]] = arith.constant 0 : index
// CHECK:           %[[PV:.*]] = ub.poison : index
// CHECK:           %[[MASK:.*]] = vector.create_mask %[[D0]] : vector<[4]xi1>
// TODO: This xfer_read is not used - avoid creating it.
// CHECK:           %[[READ:.*]] = vector.mask %[[MASK]] { vector.transfer_read %[[DEST]]{{\[}}%[[C0_1]]], %[[PV]] {in_bounds = [true]} : tensor<?xindex>, vector<[4]xindex> } : vector<[4]xi1> -> vector<[4]xindex>
// CHECK:           %[[STEP:.*]] = vector.step : vector<[4]xindex>
// CHECK:           %[[C0_3:.*]] = arith.constant 0 : index
// CHECK:           %[[WRITE:.*]] = vector.mask %[[MASK]] { vector.transfer_write %[[STEP]], %[[DEST]]{{\[}}%[[C0_3]]] {in_bounds = [true]} : vector<[4]xindex>, tensor<?xindex> } : vector<[4]xi1> -> tensor<?xindex>
// CHECK:           return %[[WRITE]] : tensor<?xindex>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [[4]] : !transform.any_op

    transform.yield
  }
}

// -----

///----------------------------------------------------------------------------------------
/// Tests for linalg.mmt4d
///----------------------------------------------------------------------------------------

func.func @mmt4d(%A: memref<16x16x8x1xf32>, %B: memref<16x16x8x1xf32>, %C_in: memref<16x16x8x8xf32>) {
  linalg.mmt4d ins(%A, %B: memref<16x16x8x1xf32>, memref<16x16x8x1xf32>)
               outs(%C_in: memref<16x16x8x8xf32>)
  return
}

// CHECK-LABEL:   func.func @mmt4d(
// CHECK-SAME:      %[[A:.*]]: memref<16x16x8x1xf32>, %[[B:.*]]: memref<16x16x8x1xf32>, %[[C:.*]]: memref<16x16x8x8xf32>) {
// CHECK:           %[[VEC_A:.*]] = vector.transfer_read %[[A]]{{.*}} : memref<16x16x8x1xf32>, vector<16x16x16x8x8x1xf32>
// CHECK:           %[[VEC_B:.*]] = vector.transfer_read %[[B]]{{.*}} : memref<16x16x8x1xf32>, vector<16x16x16x8x8x1xf32>
// CHECK:           %[[VEC_C:.*]] = vector.transfer_read %[[C]]{{.*}} : memref<16x16x8x8xf32>, vector<16x16x8x8xf32>
// CHECK:           %[[MUL:.*]] = arith.mulf %[[VEC_A]], %[[VEC_B]] : vector<16x16x16x8x8x1xf32>
// CHECK:           %[[RED:.*]] = vector.multi_reduction <add>, %[[MUL]], %[[VEC_C]] [2, 5] : vector<16x16x16x8x8x1xf32> to vector<16x16x8x8xf32>
// CHECK:           vector.transfer_write %[[RED]], %[[C]]{{.*}} : vector<16x16x8x8xf32>, memref<16x16x8x8xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %mmt4d = transform.structured.match ops{["linalg.mmt4d"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %mmt4d : !transform.any_op
    transform.yield
  }
}

// -----

func.func @mmt4d_scalable(%A: memref<16x16x8x1xf32>, %B: memref<16x16x?x1xf32>, %C_in: memref<16x16x8x?xf32>) {
  linalg.mmt4d ins(%A, %B: memref<16x16x8x1xf32>, memref<16x16x?x1xf32>)
               outs(%C_in: memref<16x16x8x?xf32>)
  return
}
// CHECK-LABEL:   func.func @mmt4d_scalable(
// CHECK-SAME:      %[[A:.*]]: memref<16x16x8x1xf32>,
// CHECK-SAME:      %[[B:.*]]: memref<16x16x?x1xf32>,
// CHECK-SAME:      %[[C_IN:.*]]: memref<16x16x8x?xf32>) {
// CHECK:           %[[C16_M:.*]] = arith.constant 16 : index
// CHECK:           %[[C16_N:.*]] = arith.constant 16 : index
// CHECK:           %[[C16_K:.*]] = arith.constant 16 : index
// CHECK:           %[[C8:.*]] = arith.constant 8 : index
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[DIM_2:.*]] = memref.dim %[[B]], %[[C2]] : memref<16x16x?x1xf32>
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[VEC_A:.*]] = vector.transfer_read %[[A]]{{.*}} : memref<16x16x8x1xf32>, vector<16x16x16x8x[4]x1xf32>
// CHECK:           %[[MASK_1:.*]] = vector.create_mask %[[C16_N]], %[[C16_K]], %[[DIM_2]], %[[C1]] : vector<16x16x[4]x1xi1>
// CHECK:           %[[VEC_B:.*]] = vector.mask %[[MASK_1]] { vector.transfer_read %[[B]]{{.*}} : memref<16x16x?x1xf32>, vector<16x16x16x8x[4]x1xf32> } : vector<16x16x[4]x1xi1> -> vector<16x16x16x8x[4]x1xf32>
// CHECK:           %[[MASK_2:.*]] = vector.create_mask %[[C16_M]], %[[C16_N]], %[[C8]], %[[DIM_2]] : vector<16x16x8x[4]xi1>
// CHECK:           %[[VEC_C:.*]] = vector.mask %[[MASK_2]] { vector.transfer_read %[[C_IN]]{{.*}} : memref<16x16x8x?xf32>, vector<16x16x8x[4]xf32> } : vector<16x16x8x[4]xi1> -> vector<16x16x8x[4]xf32>
// CHECK:           %[[MUL:.*]] = arith.mulf %[[VEC_A]], %[[VEC_B]] : vector<16x16x16x8x[4]x1xf32>
// CHECK:           %[[MASK_3:.*]] = vector.create_mask %[[C16_M]], %[[C16_N]], %[[C16_K]], %[[C8]], %[[DIM_2]], %[[C1]] : vector<16x16x16x8x[4]x1xi1>
// CHECK:           %[[RED:.*]] = vector.mask %[[MASK_3]] { vector.multi_reduction <add>, %[[MUL]], %[[VEC_C]] [2, 5] : vector<16x16x16x8x[4]x1xf32> to vector<16x16x8x[4]xf32> } : vector<16x16x16x8x[4]x1xi1> -> vector<16x16x8x[4]xf32>
// CHECK:           vector.mask %[[MASK_2]] { vector.transfer_write %[[RED]], %[[C_IN]]{{.*}} : vector<16x16x8x[4]xf32>, memref<16x16x8x?xf32> } : vector<16x16x8x[4]xi1>


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %mmt4d = transform.structured.match ops{["linalg.mmt4d"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %mmt4d vector_sizes [16, 16, 16, 8, [4], 1] : !transform.any_op
    transform.yield
  }
}

// -----

func.func @mmt4d_scalable_with_assume(%A: memref<16x16x8x1xf32>, %B: memref<16x16x?x1xf32>, %C_in: memref<16x16x8x?xf32>) {
  linalg.mmt4d ins(%A, %B: memref<16x16x8x1xf32>, memref<16x16x?x1xf32>)
               outs(%C_in: memref<16x16x8x?xf32>)
  return
}
// CHECK-LABEL:   func.func @mmt4d_scalable_with_assume(
// CHECK-SAME:      %[[A:.*]]: memref<16x16x8x1xf32>,
// CHECK-SAME:      %[[B:.*]]: memref<16x16x?x1xf32>,
// CHECK-SAME:      %[[C_IN:.*]]: memref<16x16x8x?xf32>) {
// CHECK-NOT:       mask
// CHECK:           %[[VEC_A:.*]] = vector.transfer_read %[[A]]{{.*}} : memref<16x16x8x1xf32>, vector<16x16x16x8x[4]x1xf32>
// CHECK:           %[[VEC_B:.*]] = vector.transfer_read %[[B]]{{.*}} : memref<16x16x?x1xf32>, vector<16x16x16x8x[4]x1xf32>
// CHECK:           %[[VEC_C:.*]] = vector.transfer_read %[[C_IN]]{{.*}} : memref<16x16x8x?xf32>, vector<16x16x8x[4]xf32>
// CHECK:           %[[MUL:.*]] = arith.mulf %[[VEC_A]], %[[VEC_B]] : vector<16x16x16x8x[4]x1xf32>
// CHECK:           %[[RED:.*]] = vector.multi_reduction <add>, %[[MUL]], %[[VEC_C]] [2, 5] : vector<16x16x16x8x[4]x1xf32> to vector<16x16x8x[4]xf32>
// CHECK:           vector.transfer_write %[[RED]], %[[C_IN]]{{.*}} : vector<16x16x8x[4]xf32>, memref<16x16x8x?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %mmt4d = transform.structured.match ops{["linalg.mmt4d"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %mmt4d vector_sizes [16, 16, 16, 8, [4], 1] {assume_dynamic_dims_match_vec_sizes} : !transform.any_op
    transform.yield
  }
}

// -----

///----------------------------------------------------------------------------------------
/// Tests for linalg.batch_mmt4d
///----------------------------------------------------------------------------------------

func.func @batch_mmt4d(%A: memref<2x16x16x8x1xf32>, %B: memref<2x16x16x8x1xf32>, %C_in: memref<2x16x16x8x8xf32>) {
  linalg.batch_mmt4d ins(%A, %B: memref<2x16x16x8x1xf32>, memref<2x16x16x8x1xf32>)
               outs(%C_in: memref<2x16x16x8x8xf32>)
  return
}

// CHECK-LABEL:   func.func @batch_mmt4d(
// CHECK-SAME:      %[[A:.*]]: memref<2x16x16x8x1xf32>, %[[B:.*]]: memref<2x16x16x8x1xf32>, %[[C:.*]]: memref<2x16x16x8x8xf32>) {
// CHECK:           %[[VEC_A:.*]] = vector.transfer_read %[[A]]{{.*}} : memref<2x16x16x8x1xf32>, vector<2x16x16x16x8x8x1xf32>
// CHECK:           %[[VEC_B:.*]] = vector.transfer_read %[[B]]{{.*}} : memref<2x16x16x8x1xf32>, vector<2x16x16x16x8x8x1xf32>
// CHECK:           %[[VEC_C:.*]] = vector.transfer_read %[[C]]{{.*}} : memref<2x16x16x8x8xf32>, vector<2x16x16x8x8xf32>
// CHECK:           %[[MUL:.*]] = arith.mulf %[[VEC_A]], %[[VEC_B]] : vector<2x16x16x16x8x8x1xf32>
// CHECK:           %[[RED:.*]] = vector.multi_reduction <add>, %[[MUL]], %[[VEC_C]] [3, 6] : vector<2x16x16x16x8x8x1xf32> to vector<2x16x16x8x8xf32>
// CHECK:           vector.transfer_write %[[RED]], %[[C]]{{.*}} : vector<2x16x16x8x8xf32>, memref<2x16x16x8x8xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %batch_mmt4d = transform.structured.match ops{["linalg.batch_mmt4d"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %batch_mmt4d : !transform.any_op
    transform.yield
  }
}

// -----

func.func @batch_mmt4d_scalable(%A: memref<2x16x16x8x1xf32>, %B: memref<2x16x16x?x1xf32>, %C_in: memref<2x16x16x8x?xf32>) {
  linalg.batch_mmt4d ins(%A, %B: memref<2x16x16x8x1xf32>, memref<2x16x16x?x1xf32>)
               outs(%C_in: memref<2x16x16x8x?xf32>)
  return
}
// CHECK-LABEL:   func.func @batch_mmt4d_scalable(
// CHECK-SAME:      %[[A:.*]]: memref<2x16x16x8x1xf32>,
// CHECK-SAME:      %[[B:.*]]: memref<2x16x16x?x1xf32>,
// CHECK-SAME:      %[[C_IN:.*]]: memref<2x16x16x8x?xf32>) {
// CHECK:           %[[C2:.*]] = arith.constant 2 : index
// CHECK:           %[[C16_M:.*]] = arith.constant 16 : index
// CHECK:           %[[C16_N:.*]] = arith.constant 16 : index
// CHECK:           %[[C16_K:.*]] = arith.constant 16 : index
// CHECK:           %[[C8:.*]] = arith.constant 8 : index
// CHECK:           %[[C3:.*]] = arith.constant 3 : index
// CHECK:           %[[DIM_N_IN:.*]] = memref.dim %[[B]], %[[C3]] : memref<2x16x16x?x1xf32>
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[VEC_A:.*]] = vector.transfer_read %[[A]]{{.*}} : memref<2x16x16x8x1xf32>, vector<2x16x16x16x8x[4]x1xf32>
// CHECK:           %[[MASK_1:.*]] = vector.create_mask %[[C2]], %[[C16_N]], %[[C16_K]], %[[DIM_N_IN]], %[[C1]] : vector<2x16x16x[4]x1xi1>
// CHECK:           %[[VEC_B:.*]] = vector.mask %[[MASK_1]] { vector.transfer_read %[[B]]{{.*}} : memref<2x16x16x?x1xf32>, vector<2x16x16x16x8x[4]x1xf32> } : vector<2x16x16x[4]x1xi1> -> vector<2x16x16x16x8x[4]x1xf32>
// CHECK:           %[[MASK_2:.*]] = vector.create_mask %[[C2]], %[[C16_M]], %[[C16_N]], %[[C8]], %[[DIM_N_IN]] : vector<2x16x16x8x[4]xi1>
// CHECK:           %[[VEC_C:.*]] = vector.mask %[[MASK_2]] { vector.transfer_read %[[C_IN]]{{.*}} : memref<2x16x16x8x?xf32>, vector<2x16x16x8x[4]xf32> } : vector<2x16x16x8x[4]xi1> -> vector<2x16x16x8x[4]xf32>
// CHECK:           %[[MUL:.*]] = arith.mulf %[[VEC_A]], %[[VEC_B]] : vector<2x16x16x16x8x[4]x1xf32>
// CHECK:           %[[MASK_3:.*]] = vector.create_mask %[[C2]], %[[C16_M]], %[[C16_N]], %[[C16_K]], %[[C8]], %[[DIM_N_IN]], %[[C1]] : vector<2x16x16x16x8x[4]x1xi1>
// CHECK:           %[[RED:.*]] = vector.mask %[[MASK_3]] { vector.multi_reduction <add>, %[[MUL]], %[[VEC_C]] [3, 6] : vector<2x16x16x16x8x[4]x1xf32> to vector<2x16x16x8x[4]xf32> } : vector<2x16x16x16x8x[4]x1xi1> -> vector<2x16x16x8x[4]xf32>
// CHECK:           vector.mask %[[MASK_2]] { vector.transfer_write %[[RED]], %[[C_IN]]{{.*}} : vector<2x16x16x8x[4]xf32>, memref<2x16x16x8x?xf32> } : vector<2x16x16x8x[4]xi1>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %batch_mmt4d = transform.structured.match ops{["linalg.batch_mmt4d"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %batch_mmt4d vector_sizes [2, 16, 16, 16, 8, [4], 1] : !transform.any_op
    transform.yield
  }
}

// -----

func.func @batch_mmt4d_scalable_with_assume(%A: memref<2x16x16x8x1xf32>, %B: memref<2x16x16x?x1xf32>, %C_in: memref<2x16x16x8x?xf32>) {
  linalg.batch_mmt4d ins(%A, %B: memref<2x16x16x8x1xf32>, memref<2x16x16x?x1xf32>)
               outs(%C_in: memref<2x16x16x8x?xf32>)
  return
}
// CHECK-LABEL:   func.func @batch_mmt4d_scalable_with_assume(
// CHECK-SAME:      %[[A:.*]]: memref<2x16x16x8x1xf32>,
// CHECK-SAME:      %[[B:.*]]: memref<2x16x16x?x1xf32>,
// CHECK-SAME:      %[[C_IN:.*]]: memref<2x16x16x8x?xf32>) {
// CHECK-NOT:       mask
// CHECK:           %[[VEC_A:.*]] = vector.transfer_read %[[A]]{{.*}} : memref<2x16x16x8x1xf32>, vector<2x16x16x16x8x[4]x1xf32>
// CHECK:           %[[VEC_B:.*]] = vector.transfer_read %[[B]]{{.*}} : memref<2x16x16x?x1xf32>, vector<2x16x16x16x8x[4]x1xf32>
// CHECK:           %[[VEC_C:.*]] = vector.transfer_read %[[C_IN]]{{.*}} : memref<2x16x16x8x?xf32>, vector<2x16x16x8x[4]xf32>
// CHECK:           %[[MUL:.*]] = arith.mulf %[[VEC_A]], %[[VEC_B]] : vector<2x16x16x16x8x[4]x1xf32>
// CHECK:           %[[RED:.*]] = vector.multi_reduction <add>, %[[MUL]], %[[VEC_C]] [3, 6] : vector<2x16x16x16x8x[4]x1xf32> to vector<2x16x16x8x[4]xf32>
// CHECK:           vector.transfer_write %[[RED]], %[[C_IN]]{{.*}} : vector<2x16x16x8x[4]xf32>, memref<2x16x16x8x?xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %batch_mmt4d = transform.structured.match ops{["linalg.batch_mmt4d"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %batch_mmt4d vector_sizes [2, 16, 16, 16, 8, [4], 1] {assume_dynamic_dims_match_vec_sizes} : !transform.any_op
    transform.yield
  }
}


// -----

///----------------------------------------------------------------------------------------
/// Tests for linalg.unpack
///----------------------------------------------------------------------------------------

// CHECK-LABEL: func @test_vectorize_dynamic_shapes_unpack
// CHECK-SAME:      %[[DEST:.*]]: tensor<?x?xf32>,
// CHECK-SAME:      %[[SRC:.*]]: tensor<?x?x16x2xf32>
func.func @test_vectorize_dynamic_shapes_unpack(%dest: tensor<?x?xf32>, %src: tensor<?x?x16x2xf32>) -> tensor<?x?xf32> {
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[C0_1:.*]] = arith.constant 0 : index
  // CHECK: %[[DIM_0:.*]] = tensor.dim %[[SRC]], %[[C0_1]] : tensor<?x?x16x2xf32>
  // CHECK: %[[C1:.*]] = arith.constant 1
  // CHECK: %[[DIM6:.*]] = tensor.dim %[[SRC]], %[[C1]] : tensor<?x?x16x2xf32>
  // CHECK: %[[CNST16:.*]] = arith.constant 16 : index
  // CHECK: %[[CNST2:.*]] = arith.constant 2 : index
  // CHECK: %[[MASK_READ:.*]] = vector.create_mask %[[DIM_0]], %[[DIM6]], %[[CNST16]], %[[CNST2]] : vector<2x1x16x2xi1>
  // CHECK: %[[READ:.*]] = vector.mask %[[MASK_READ]] {{.*}} vector.transfer_read %{{.*}} : tensor<?x?x16x2xf32>, vector<2x1x16x2xf32> } : vector<2x1x16x2xi1> -> vector<2x1x16x2xf32>
  // CHECK: %[[TR:.*]] = vector.transpose %[[READ]], [0, 3, 1, 2] : vector<2x1x16x2xf32> to vector<2x2x1x16xf32>
  // CHECK: %[[SC:.*]] = vector.shape_cast %[[TR]] : vector<2x2x1x16xf32> to vector<4x16xf32>
  // CHECK: %[[MASK_WRITE:.*]] = vector.create_mask {{.*}} : vector<4x16xi1>
  // CHECK: %[[WRITE:.*]] = vector.mask %[[MASK_WRITE:.*]] {{.*}} vector.transfer_write %[[SC]], %[[DEST]]
  // CHECK: return %[[WRITE]]
  %ret = linalg.unpack %src inner_dims_pos = [1, 0] inner_tiles = [16, 2] into %dest : tensor<?x?x16x2xf32> -> tensor<?x?xf32>
  return %ret : tensor<?x?xf32>
}
module attributes {transform.with_named_sequence} {
 transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
   %0 = transform.structured.match ops{["linalg.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
   transform.structured.vectorize %0 vector_sizes [2, 1, 16, 2] : !transform.any_op
   transform.yield
 }
}

// -----

// CHECK-LABEL: func @test_vectorize_dynamic_shapes_unpack_scalable_vec
// CHECK-SAME:      %[[DEST:.*]]: tensor<?x?xf32>,
// CHECK-SAME:      %[[SRC:.*]]: tensor<?x?x16x2xf32>
func.func @test_vectorize_dynamic_shapes_unpack_scalable_vec(%dest: tensor<?x?xf32>, %src: tensor<?x?x16x2xf32>) -> tensor<?x?xf32> {
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00
  // CHECK: %[[C01:.*]] = arith.constant 0
  // CHECK: %[[C02:.*]] = arith.constant 0
  // CHECK: %[[DIM4:.*]] = tensor.dim %[[SRC]], %[[C02]] : tensor<?x?x16x2xf32>
  // CHECK: %[[CNST14:.*]] = arith.constant 1
  // CHECK: %[[DIM6:.*]] = tensor.dim %[[SRC]], %[[CNST14]] : tensor<?x?x16x2xf32>
  // CHECK: %[[CNST16:.*]] = arith.constant 16 : index
  // CHECK: %[[CNST2:.*]] = arith.constant 2 : index
  // CHECK: %[[MASK_READ:.*]] = vector.create_mask %[[DIM4]], %[[DIM6]], %[[CNST16]], %[[CNST2]] : vector<2x1x[16]x2xi1>
  // CHECK: %[[READ:.*]] = vector.mask %[[MASK_READ]] {{.*}} vector.transfer_read %{{.*}} : tensor<?x?x16x2xf32>, vector<2x1x[16]x2xf32> } : vector<2x1x[16]x2xi1> -> vector<2x1x[16]x2xf32>
  // CHECK: %[[TR:.*]] = vector.transpose %[[READ]], [0, 3, 1, 2] : vector<2x1x[16]x2xf32> to vector<2x2x1x[16]xf32>
  // CHECK: %[[SC:.*]] = vector.shape_cast %[[TR]] : vector<2x2x1x[16]xf32> to vector<4x[16]xf32>
  // CHECK: %[[MASK_WRITE:.*]] = vector.create_mask {{.*}} : vector<4x[16]xi1>
  // CHECK: %[[WRITE:.*]] = vector.mask %[[MASK_WRITE:.*]] {{.*}} vector.transfer_write %[[SC]], %[[DEST]]
  // CHECK: return %[[WRITE]]
  %ret = linalg.unpack %src inner_dims_pos = [1, 0] inner_tiles = [16, 2] into %dest : tensor<?x?x16x2xf32> -> tensor<?x?xf32>
  return %ret : tensor<?x?xf32>
}
module attributes {transform.with_named_sequence} {
 transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
   %0 = transform.structured.match ops{["linalg.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
   transform.structured.vectorize %0 vector_sizes [2, 1, [16], 2] : !transform.any_op
   transform.yield
 }
}

// -----

// CHECK-LABEL: func @test_vectorize_dynamic_shapes_unpack_scalable_vec_and_tile_size
// CHECK-SAME:      %[[DEST:.*]]: tensor<?x?xf32>,
// CHECK-SAME:      %[[SRC:.*]]: tensor<?x?x?x2xf32>
func.func @test_vectorize_dynamic_shapes_unpack_scalable_vec_and_tile_size(%dest: tensor<?x?xf32>, %src: tensor<?x?x?x2xf32>) -> tensor<?x?xf32> {
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00
  // CHECK: %[[C01:.*]] = arith.constant 0
  // CHECK: %[[C02:.*]] = arith.constant 0
  // CHECK: %[[DIM4:.*]] = tensor.dim %[[SRC]], %[[C02]] : tensor<?x?x?x2xf32>
  // CHECK: %[[C1_2:.*]] = arith.constant 1
  // CHECK: %[[DIM6:.*]] = tensor.dim %[[SRC]], %[[C1_2]] : tensor<?x?x?x2xf32>
  // CHECK: %[[C2:.*]] = arith.constant 2 : index
  // CHECK: %[[DIM_2:.*]] = tensor.dim %[[SRC]], %[[C2]] : tensor<?x?x?x2xf32>
  // CHECK: %[[C2_1:.*]] = arith.constant 2 : index
  // CHECK: %[[MASK_READ:.*]] = vector.create_mask %[[DIM4]], %[[DIM6]], %[[DIM_2]], %[[C2_1]] : vector<2x1x[16]x2xi1>
  // CHECK: %[[READ:.*]] = vector.mask %[[MASK_READ]] {{.*}} vector.transfer_read %{{.*}} : tensor<?x?x?x2xf32>, vector<2x1x[16]x2xf32> } : vector<2x1x[16]x2xi1> -> vector<2x1x[16]x2xf32>
  // CHECK: %[[TR:.*]] = vector.transpose %[[READ]], [0, 3, 1, 2] : vector<2x1x[16]x2xf32> to vector<2x2x1x[16]xf32>
  // CHECK: %[[SC:.*]] = vector.shape_cast %[[TR]] : vector<2x2x1x[16]xf32> to vector<4x[16]xf32>
  // CHECK: %[[MASK_WRITE:.*]] = vector.create_mask {{.*}} : vector<4x[16]xi1>
  // CHECK: %[[WRITE:.*]] = vector.mask %[[MASK_WRITE:.*]] {{.*}} vector.transfer_write %[[SC]], %[[DEST]]
  // CHECK: return %[[WRITE]]

  %vs = vector.vscale
  %c16 = arith.constant 16 : index
  %tile_size = arith.muli %vs, %c16 : index

  %ret = linalg.unpack %src inner_dims_pos = [1, 0] inner_tiles = [%tile_size, 2] into %dest : tensor<?x?x?x2xf32> -> tensor<?x?xf32>
  return %ret : tensor<?x?xf32>
}
module attributes {transform.with_named_sequence} {
 transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
   %0 = transform.structured.match ops{["linalg.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
   transform.structured.vectorize %0 vector_sizes [2, 1, [16], 2] : !transform.any_op
   transform.yield
 }
}

// -----

// CHECK-LABEL: func @test_vectorize_unpack
// CHECK-SAME:      %[[SRC:.*]]: tensor<8x8x32x16xf32>
// CHECK-SAME:      %[[DEST:.*]]: tensor<256x128xf32>
func.func @test_vectorize_unpack(%source: tensor<8x8x32x16xf32>, %dest: tensor<256x128xf32>) -> tensor<256x128xf32> {
    // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[C0:.*]]= arith.constant 0 : index
    // CHECK: %[[C8:.*]] = arith.constant 8 : index
    // CHECK: %[[C80:.*]] = arith.constant 8 : index
    // CHECK: %[[C32:.*]] = arith.constant 32 : index
    // CHECK: %[[C16:.*]] = arith.constant 16 : index
    // CHECK: %[[MSK0:.*]] = vector.create_mask %[[C8]], %[[C80]], %[[C32]], %[[C16]] : vector<16x8x32x16xi1>
    // CHECK: %[[READ0:.*]] = vector.mask %[[MSK0]] { vector.transfer_read %[[SRC]]{{.*}}} : vector<16x8x32x16xi1> -> vector<16x8x32x16xf32>
    // CHECK: %[[TRANSP0:.*]] = vector.transpose %[[READ0]], [0, 2, 1, 3] : vector<16x8x32x16xf32> to vector<16x32x8x16xf32>
    // CHECK: %[[SHAPC:.*]] = vector.shape_cast %[[TRANSP0]] : vector<16x32x8x16xf32> to vector<512x128xf32>
    // CHECK: %[[C01:.*]] = arith.constant 0 : index
    // CHECK: %[[C256:.*]] = arith.constant 256 : index
    // CHECK: %[[C128:.*]] = arith.constant 128 : index
    // CHECK: %[[WRITEMSK:.*]] = vector.create_mask %[[C256]], %[[C128]] : vector<512x128xi1>
    // CHECK: %[[WRIT:.*]] = vector.mask %[[WRITEMSK]] { vector.transfer_write %[[SHAPC]], %[[DEST]]{{.*}}} : vector<512x128xi1> -> tensor<256x128xf32>
    // CHECK: return %[[WRIT]] : tensor<256x128xf32>
   %0 = linalg.unpack %source inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %dest : tensor<8x8x32x16xf32> -> tensor<256x128xf32>
   return %0 : tensor<256x128xf32>
 }
 module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
   transform.structured.vectorize %0 vector_sizes [16, 8, 32, 16] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @test_vectorize_unpack_no_masks
// CHECK-SAME:      %[[SRC:.*]]: tensor<8x8x32x16xf32>
// CHECK-SAME:      %[[DEST:.*]]: tensor<256x128xf32>
func.func @test_vectorize_unpack_no_masks(%source: tensor<8x8x32x16xf32>, %dest: tensor<256x128xf32>) -> tensor<256x128xf32> {
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[READ:.*]] = vector.transfer_read %[[SRC]]{{.*}}} : tensor<8x8x32x16xf32>, vector<8x8x32x16xf32> 
  // CHECK: %[[TRANSP:.*]] = vector.transpose %[[READ]], [0, 2, 1, 3] : vector<8x8x32x16xf32> to vector<8x32x8x16xf32>
  // CHECK: %[[SHAPC:.*]] = vector.shape_cast %[[TRANSP]] : vector<8x32x8x16xf32> to vector<256x128xf32>
  // CHECK: %[[C00:.*]] = arith.constant 0 : index
  // CHECK: %[[WRIT:.*]] = vector.transfer_write %[[SHAPC]], %[[DEST]]{{.*}}} : vector<256x128xf32>, tensor<256x128xf32> 
  // CHECK: return %[[WRIT]] : tensor<256x128xf32>
   %0 = linalg.unpack %source inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %dest : tensor<8x8x32x16xf32> -> tensor<256x128xf32>
   return %0 : tensor<256x128xf32>
 }
 module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
   transform.structured.vectorize %0 vector_sizes [8, 8, 32, 16] : !transform.any_op
    transform.yield
  }
 }

// -----

// CHECK-LABEL: test_vectorize_unpack_with_outer_perm
// CHECK-SAME:      %[[SRC:.*]]: tensor<8x8x32x16xf32>
// CHECK-SAME:      %[[DEST:.*]]: tensor<256x128xf32>
  func.func @test_vectorize_unpack_with_outer_perm(%source: tensor<8x8x32x16xf32>, %dest: tensor<256x128xf32>) -> tensor<256x128xf32> {
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[READ:.*]] = vector.transfer_read %[[SRC]]{{.*}}} : tensor<8x8x32x16xf32>, vector<8x8x32x16xf32> 
  // CHECK: %[[TRANSP:.*]] = vector.transpose %[[READ]], [1, 2, 0, 3] : vector<8x8x32x16xf32> to vector<8x32x8x16xf32>
  // CHECK: %[[SHAPC:.*]] = vector.shape_cast %[[TRANSP]] : vector<8x32x8x16xf32> to vector<256x128xf32>
  // CHECK: %[[C00:.*]] = arith.constant 0 : index
  // CHECK: %[[WRIT:.*]] = vector.transfer_write %[[SHAPC]], %[[DEST]]{{.*}}} : vector<256x128xf32>, tensor<256x128xf32> 
  // CHECK: return %[[WRIT]] : tensor<256x128xf32>
   %0 = linalg.unpack %source outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %dest : tensor<8x8x32x16xf32> -> tensor<256x128xf32>
   return %0 : tensor<256x128xf32>
 }
 module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
   transform.structured.vectorize %0 vector_sizes [8, 8, 32, 16] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: test_vectorize_unpack_no_vector_sizes
// CHECK-SAME:      %[[SRC:.*]]: tensor<8x8x32x16xf32>
// CHECK-SAME:      %[[DEST:.*]]: tensor<256x128xf32>
func.func @test_vectorize_unpack_no_vector_sizes(%source: tensor<8x8x32x16xf32>, %dest: tensor<256x128xf32>) -> tensor<256x128xf32> {
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[READ:.*]] = vector.transfer_read %[[SRC]]{{.*}}} : tensor<8x8x32x16xf32>, vector<8x8x32x16xf32> 
  // CHECK: %[[TRANSP:.*]] = vector.transpose %[[READ]], [0, 2, 1, 3] : vector<8x8x32x16xf32> to vector<8x32x8x16xf32>
  // CHECK: %[[SHAPC:.*]] = vector.shape_cast %[[TRANSP]] : vector<8x32x8x16xf32> to vector<256x128xf32>
  // CHECK: %[[C00:.*]] = arith.constant 0 : index
  // CHECK: %[[WRIT:.*]] = vector.transfer_write %[[SHAPC]], %[[DEST]]{{.*}}} : vector<256x128xf32>, tensor<256x128xf32> 
  // CHECK: return %[[WRIT]] : tensor<256x128xf32>
   %0 = linalg.unpack %source inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %dest : tensor<8x8x32x16xf32> -> tensor<256x128xf32>
   return %0 : tensor<256x128xf32>
 }
 module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
   transform.structured.vectorize %0 : !transform.any_op
    transform.yield
  }
 }

// -----

// CHECK-LABEL: test_vectorize_unpack_no_vector_sizes_slice_output
// CHECK-SAME:      %[[SRC:.*]]: tensor<8x4x16x16xf32>
// CHECK-SAME:      %[[DEST:.*]]: tensor<64x127xf32>
func.func @test_vectorize_unpack_no_vector_sizes_slice_output(%source: tensor<8x4x16x16xf32>, %dest: tensor<64x127xf32>) -> tensor<64x127xf32> {
  //      CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  //      CHECK: %[[C0:.*]] = arith.constant 0 : index
  //      CHECK: %[[READ:.*]] = vector.transfer_read %[[SRC]]{{.*}}} : tensor<8x4x16x16xf32>, vector<8x4x16x16xf32>
  //      CHECK: %[[TRANSP:.*]] = vector.transpose %[[READ]], [1, 2, 0, 3] : vector<8x4x16x16xf32> to vector<4x16x8x16xf32>
  //      CHECK: %[[SHAPC:.*]] = vector.shape_cast %[[TRANSP]] : vector<4x16x8x16xf32> to vector<64x128xf32>
  //      CHECK: %[[C00:.*]] = arith.constant 0 : index
  //      CHECK: %[[WRIT:.*]] = vector.transfer_write %[[SHAPC]], %[[DEST]]
  // CHECK-SAME:  {in_bounds = [true, false]} : vector<64x128xf32>, tensor<64x127xf32>
  //      CHECK: return %[[WRIT]] : tensor<64x127xf32>
   %0 = linalg.unpack %source outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %dest : tensor<8x4x16x16xf32> -> tensor<64x127xf32>
   return %0 : tensor<64x127xf32>
 }
 module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
   transform.structured.vectorize %0 : !transform.any_op
    transform.yield
  }
 }

// -----

// CHECK-LABEL: test_vectorize_unpack_no_vector_sizes_permute
// CHECK-SAME:      %[[SRC:.*]]:  tensor<4x7x4xf32>
// CHECK-SAME:      %[[DEST:.*]]:  tensor<7x16xf32>
func.func @test_vectorize_unpack_no_vector_sizes_permute(%source: tensor<4x7x4xf32>, %dest: tensor<7x16xf32>) -> tensor<7x16xf32> {
   %0 = linalg.unpack %source outer_dims_perm=[1, 0] inner_dims_pos = [1] inner_tiles = [4] into %dest : tensor<4x7x4xf32> -> tensor<7x16xf32>
   return %0 : tensor<7x16xf32>
 }
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[READ:.*]] = vector.transfer_read %[[SRC]]{{.*}}} : tensor<4x7x4xf32>, vector<4x7x4xf32>
  // CHECK: %[[TRANSP:.*]] = vector.transpose %[[READ]], [1, 0, 2] : vector<4x7x4xf32> to vector<7x4x4xf32>
  // CHECK: %[[SHAPC:.*]] = vector.shape_cast %[[TRANSP]] : vector<7x4x4xf32> to vector<7x16xf32>
  // CHECK: %[[C00:.*]] = arith.constant 0 : index
  // CHECK: %[[WRIT:.*]] = vector.transfer_write %[[SHAPC]], %[[DEST]]{{.*}}} : vector<7x16xf32>, tensor<7x16xf32> 
  // CHECK: return %[[WRIT]] : tensor<7x16xf32>
 module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
   transform.structured.vectorize %0 : !transform.any_op
    transform.yield
  }
 }

// -----

///----------------------------------------------------------------------------------------
/// Tests for linalg.pack
///----------------------------------------------------------------------------------------

// Input identical as the test in vectorization-with-patterns.mlir. Output is
// different - vector sizes are inferred (rather than user-specified) and hence
// masking was used.

func.func @test_vectorize_pack(%arg0: tensor<32x8x16xf32>, %arg1: tensor<4x1x32x16x2xf32>) -> tensor<4x1x32x16x2xf32> {
  %pack = linalg.pack %arg0 outer_dims_perm = [1, 2, 0] inner_dims_pos = [2, 1] inner_tiles = [16, 2] into %arg1 : tensor<32x8x16xf32> -> tensor<4x1x32x16x2xf32>
  return %pack : tensor<4x1x32x16x2xf32>
}
//  CHECK-DAG: %[[cst:.*]] = arith.constant 0.000000e+00 : f32
//  CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
//      CHECK: %[[read:.*]] = vector.transfer_read %{{.*}}[%[[c0]], %[[c0]], %[[c0]]], %[[cst]]
// CHECK-SAME:    {in_bounds = [true, true, true]} : tensor<32x8x16xf32>, vector<32x8x16xf32>
//      CHECK: %[[shape_cast:.*]] = vector.shape_cast %[[read]] : vector<32x8x16xf32> to vector<32x4x2x1x16xf32>
//      CHECK: %[[transpose:.*]] = vector.transpose %[[shape_cast]], [1, 3, 0, 4, 2] : vector<32x4x2x1x16xf32> to vector<4x1x32x16x2xf32>
//  CHECK-DAG: %[[c0_1:.*]] = arith.constant 0 : index
//  CHECK-DAG: %[[empty:.*]] = tensor.empty() : tensor<4x1x32x16x2xf32>
//      CHECK: %[[write:.*]] = vector.transfer_write %[[transpose]], %[[empty]][%[[c0_1]], %[[c0_1]], %[[c0_1]], %[[c0_1]], %[[c0_1]]]
// CHECK-SAME:   {in_bounds = [true, true, true, true, true]} : vector<4x1x32x16x2xf32>, tensor<4x1x32x16x2xf32>
//      CHECK: return %[[write]] : tensor<4x1x32x16x2xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.pack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [4, 1, 32] : !transform.any_op
    transform.yield
  }
}

// -----

// Input identical as the test in vectorization-with-patterns.mlir. Output is
// different - vector sizes are inferred (rather than user-specified) and hence
// masking was used.

func.func @test_vectorize_padded_pack(%arg0: tensor<32x7x15xf32>, %arg1: tensor<32x4x1x16x2xf32>) -> tensor<32x4x1x16x2xf32> {
  %pad = arith.constant 0.000000e+00 : f32
  %pack = linalg.pack %arg0 padding_value(%pad : f32) inner_dims_pos = [2, 1] inner_tiles = [16, 2] into %arg1 : tensor<32x7x15xf32> -> tensor<32x4x1x16x2xf32>
  return %pack : tensor<32x4x1x16x2xf32>
}
//  CHECK-DAG: %[[cst:.*]] = arith.constant 0.000000e+00 : f32
//  CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
//  CHECK-DAG: %[[c32:.*]] = arith.constant 32 : index
//  CHECK-DAG: %[[c7:.*]] = arith.constant 7 : index
//  CHECK-DAG: %[[c15:.*]] = arith.constant 15 : index
//      CHECK: %[[mask:.*]] = vector.create_mask %[[c32]], %[[c7]], %[[c15]] : vector<32x8x16xi1>
//      CHECK: %[[masked_read:.*]] = vector.mask %[[mask]] {
// CHECK-SAME:   vector.transfer_read %{{.*}}[%[[c0]], %[[c0]], %[[c0]]], %[[cst]]
// CHECK-SAME:   {in_bounds = [true, true, true]} : tensor<32x7x15xf32>, vector<32x8x16xf32>
// CHECK-SAME: } : vector<32x8x16xi1> -> vector<32x8x16xf32>
//      CHECK: %[[shape_cast:.*]] = vector.shape_cast %[[masked_read]] : vector<32x8x16xf32> to vector<32x4x2x1x16xf32>
//      CHECK: %[[transpose:.*]] = vector.transpose %[[shape_cast]], [0, 1, 3, 4, 2] : vector<32x4x2x1x16xf32> to vector<32x4x1x16x2xf32>
//  CHECK-DAG: %[[c0_1:.*]] = arith.constant 0 : index
//  CHECK-DAG: %[[empty:.*]] = tensor.empty() : tensor<32x4x1x16x2xf32>
//      CHECK: %[[write:.*]] = vector.transfer_write %[[transpose]], %[[empty]][%[[c0_1]], %[[c0_1]], %[[c0_1]], %[[c0_1]], %[[c0_1]]]
// CHECK-SAME:   {in_bounds = [true, true, true, true, true]} : vector<32x4x1x16x2xf32>, tensor<32x4x1x16x2xf32>
//      CHECK: return %[[write]] : tensor<32x4x1x16x2xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.pack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [32, 4, 1] : !transform.any_op
    transform.yield
  }
}

// -----

func.func @test_vectorize_dynamic_pack(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?x16x2xf32>) -> tensor<?x?x16x2xf32> {
  %pack = linalg.pack %arg0 inner_dims_pos = [1, 0] inner_tiles = [16, 2] into %arg1 : tensor<?x?xf32> -> tensor<?x?x16x2xf32>
  return %pack : tensor<?x?x16x2xf32>
}
//  CHECK-DAG: %[[cst:.*]] = arith.constant 0.000000e+00 : f32
//  CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
//  CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
//  CHECK-DAG: %[[d0:.*]] = tensor.dim {{.*}} %[[c0]] : tensor<?x?x16x2xf32>
//  CHECK-DAG: %[[d1:.*]] = tensor.dim {{.*}} %[[c1]] : tensor<?x?x16x2xf32>
//  CHECK-DAG: %[[c0_1:.*]] = arith.constant 0 : index
//  CHECK-DAG: %[[c0_0:.*]] = arith.constant 0 : index
//  CHECK-DAG: %[[c1_0:.*]] = arith.constant 1 : index
//  CHECK-DAG: %[[d0_0:.*]] = tensor.dim {{.*}} %[[c0_0]] : tensor<?x?xf32>
//  CHECK-DAG: %[[d1_0:.*]] = tensor.dim {{.*}} %[[c1_0]] : tensor<?x?xf32>
//      CHECK: %[[mask:.*]] = vector.create_mask %[[d0_0]], %[[d1_0]] : vector<8x16xi1>
//      CHECK: %[[masked_read:.*]] = vector.mask %[[mask]] {
// CHECK-SAME:   vector.transfer_read %{{.*}}[%[[c0_1]], %[[c0_1]]], %[[cst]]
// CHECK-SAME:   {in_bounds = [true, true]} : tensor<?x?xf32>, vector<8x16xf32>
// CHECK-SAME: } : vector<8x16xi1> -> vector<8x16xf32>
//      CHECK: %[[shape_cast:.*]] = vector.shape_cast %[[masked_read]] : vector<8x16xf32> to vector<4x2x1x16xf32>
//      CHECK: %[[transpose:.*]] = vector.transpose %[[shape_cast]], [0, 2, 3, 1] : vector<4x2x1x16xf32> to vector<4x1x16x2xf32>
//  CHECK-DAG: %[[c0_2:.*]] = arith.constant 0 : index
//  CHECK-DAG: %[[c16:.*]] = arith.constant 16 : index
//  CHECK-DAG: %[[c2:.*]] = arith.constant 2 : index
//  CHECK-DAG: %[[empty:.*]] = tensor.empty(%[[d0]], %[[d1]]) : tensor<?x?x16x2xf32>
//  CHECK-DAG: %[[d2:.*]] = tensor.dim %[[empty]], {{.*}} : tensor<?x?x16x2xf32>
//  CHECK-DAG: %[[d3:.*]] = tensor.dim %[[empty]], {{.*}} : tensor<?x?x16x2xf32>
//      CHECK: %[[mask_0:.*]] = vector.create_mask %[[d2]], %[[d3]], %[[c16]], %[[c2]] : vector<4x1x16x2xi1>
//      CHECK: %[[masked_write:.*]] = vector.mask %[[mask_0]] {
// CHECK-SAME:   vector.transfer_write %[[transpose]], %[[empty]][%[[c0_2]], %[[c0_2]], %[[c0_2]], %[[c0_2]]]
// CHECK-SAME:   {in_bounds = [true, true, true, true]} : vector<4x1x16x2xf32>, tensor<?x?x16x2xf32>
//      CHECK: return %[[masked_write]] : tensor<?x?x16x2xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.pack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [4, 1] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: test_vectorize_pack_no_vector_sizes
func.func @test_vectorize_pack_no_vector_sizes(%arg0: tensor<64x4xf32>, %arg1: tensor<2x4x16x2xf32>) -> tensor<2x4x16x2xf32> {
  %pack = linalg.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [16, 2] into %arg1 : tensor<64x4xf32> -> tensor<2x4x16x2xf32>
  return %pack : tensor<2x4x16x2xf32>
}
//  CHECK-DAG: %[[cst:.*]] = arith.constant 0.000000e+00 : f32
//  CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
//      CHECK: %[[read:.*]] = vector.transfer_read %{{.*}}[%[[c0]], %[[c0]]], %[[cst]]
// CHECK-SAME:    {in_bounds = [true, true]} : tensor<64x4xf32>, vector<64x4xf32>
//      CHECK: %[[shape_cast:.*]] = vector.shape_cast %[[read]] : vector<64x4xf32> to vector<4x16x2x2xf32>
//      CHECK: %[[transpose:.*]] = vector.transpose %[[shape_cast]], [2, 0, 1, 3] : vector<4x16x2x2xf32> to vector<2x4x16x2xf32>
//  CHECK-DAG: %[[c0_1:.*]] = arith.constant 0 : index
//  CHECK-DAG: %[[empty:.*]] = tensor.empty() : tensor<2x4x16x2xf32>
//      CHECK: %[[write:.*]] = vector.transfer_write %[[transpose]], %[[empty]][%[[c0_1]], %[[c0_1]], %[[c0_1]], %[[c0_1]]]
// CHECK-SAME:   {in_bounds = [true, true, true, true]} : vector<2x4x16x2xf32>, tensor<2x4x16x2xf32>
//      CHECK: return %[[write]] : tensor<2x4x16x2xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.pack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: test_vectorize_padded_pack_no_vector_sizes
func.func @test_vectorize_padded_pack_no_vector_sizes(%arg0: tensor<32x7x15xf32>, %arg1: tensor<32x4x1x16x2xf32>) -> tensor<32x4x1x16x2xf32> {
  %pad = arith.constant 0.000000e+00 : f32
  %pack = linalg.pack %arg0 padding_value(%pad : f32) inner_dims_pos = [2, 1] inner_tiles = [16, 2] into %arg1 : tensor<32x7x15xf32> -> tensor<32x4x1x16x2xf32>
  return %pack : tensor<32x4x1x16x2xf32>
}
//  CHECK-DAG: %[[cst:.*]] = arith.constant 0.000000e+00 : f32
//  CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
//      CHECK: %[[transfer_read:.*]] =  vector.transfer_read %{{.*}}[%[[c0]], %[[c0]], %[[c0]]], %[[cst]]
// CHECK-SAME:   {in_bounds = [true, false, false]} : tensor<32x7x15xf32>, vector<32x8x16xf32>
//      CHECK: %[[shape_cast:.*]] = vector.shape_cast %[[transfer_read]] : vector<32x8x16xf32> to vector<32x4x2x1x16xf32>
//      CHECK: %[[transpose:.*]] = vector.transpose %[[shape_cast]], [0, 1, 3, 4, 2] : vector<32x4x2x1x16xf32> to vector<32x4x1x16x2xf32>
//  CHECK-DAG: %[[c0_1:.*]] = arith.constant 0 : index
//  CHECK-DAG: %[[empty:.*]] = tensor.empty() : tensor<32x4x1x16x2xf32>
//      CHECK: %[[write:.*]] = vector.transfer_write %[[transpose]], %[[empty]][%[[c0_1]], %[[c0_1]], %[[c0_1]], %[[c0_1]], %[[c0_1]]]
// CHECK-SAME:   {in_bounds = [true, true, true, true, true]} : vector<32x4x1x16x2xf32>, tensor<32x4x1x16x2xf32>
//      CHECK: return %[[write]] : tensor<32x4x1x16x2xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.pack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 : !transform.any_op
    transform.yield
  }
}


///----------------------------------------------------------------------------------------
/// Tests for other Ops
///----------------------------------------------------------------------------------------

// -----

func.func @vectorize_dynamic_fill(%A : tensor<?x?xf32>, %arg0 : f32) -> tensor<?x?xf32> {
  %0 = linalg.fill ins(%arg0 : f32) outs(%A : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @vectorize_dynamic_fill
//   CHECK: %[[DIM0:.*]] = tensor.dim
//   CHECK: %[[DIM1:.*]] = tensor.dim
//   CHECK: %[[MASK:.*]] = vector.create_mask %[[DIM0]], %[[DIM1]] : vector<8x16xi1>
//   CHECK: %[[BCAST:.*]] = vector.broadcast %{{.*}} : f32 to vector<8x16xf32>
//   CHECK: vector.mask %[[MASK]] { vector.transfer_write %[[BCAST]], {{.*}} {in_bounds = [true, true]} : vector<8x16xf32>, tensor<?x?xf32> } : vector<8x16xi1>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [8, 16] : !transform.any_op
    transform.yield
  }
}

// -----

// NOTE: Often, non-trailing scalable sizes are problematic - there are no
// "scalable" arrays of vectors at the LLVM level (multi-dim vectors are
// decomposed into arrays of aggregates). However, the trailing dim in this
// case is 1 and that can be folded away later.

// NOTE: This is similar to the example above, but the trailing dim was set to
// 1 to make it foldable + vectorizable.

func.func @vectorize_dynamic_fill_scalable(%A : tensor<?x?xf32>, %arg0 : f32) -> tensor<?x?xf32> {
  %0 = linalg.fill ins(%arg0 : f32) outs(%A : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}

// CHECK-LABEL: func.func @vectorize_dynamic_fill_scalable
//   CHECK: %[[DIM0:.*]] = tensor.dim
//   CHECK: %[[DIM1:.*]] = tensor.dim
//   CHECK: %[[MASK:.*]] = vector.create_mask %[[DIM0]], %[[DIM1]] : vector<[8]x1xi1>
//   CHECK: %[[BCAST:.*]] = vector.broadcast %{{.*}} : f32 to vector<[8]x1xf32>
//   CHECK: vector.mask %[[MASK]] { vector.transfer_write %[[BCAST]], {{.*}} {in_bounds = [true, true]} : vector<[8]x1xf32>, tensor<?x?xf32> } : vector<[8]x1xi1>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [[8], 1] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK: #[[MAP:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK: func @test_masked_vectorize_linalg_transpose
func.func @test_masked_vectorize_linalg_transpose(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?xf32>) -> tensor<?x?xf32> {
  // CHECK-DAG:  %[[C0:.*]] = arith.constant 0 : index
  // CHECK-DAG:  %[[D0:.*]] = tensor.dim %arg0, %[[C0]]
  // CHECK-DAG:  %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG:  %[[D1:.*]] = tensor.dim %arg0, %[[C1]]
  // CHECK:      %[[MASK0:.*]] = vector.create_mask %[[D0]], %[[D1]]
  // CHECK:      %[[LOAD:.*]] = vector.mask %[[MASK0]] { vector.transfer_read %arg0{{.+}} permutation_map = #[[MAP]]{{.+}} }
  // CHECK-SAME:   vector<4x2xi1> -> vector<2x4xf32>
  // CHECK:      %[[MASK1:.*]] = vector.create_mask %[[D1]], %[[D0]]
  // CHECK:      %[[WRITE:.*]] = vector.mask %[[MASK1]] { vector.transfer_write %[[LOAD]], %arg1{{.+}} }
  // CHECK-SAME:   vector<2x4xi1> -> tensor<?x?xf32>
  // CHECK:      return %[[WRITE]]
  %0 = linalg.transpose ins(%arg0 : tensor<?x?xf32>) outs(%arg1 : tensor<?x?xf32>) permutation = [1, 0]
  return %0 : tensor<?x?xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.transpose"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [2, 4] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @test_masked_vectorize_linalg_copy
func.func @test_masked_vectorize_linalg_copy(%A : memref<?x?xf32>, %B : memref<?x?xf32>) {
  // CHECK: %[[c0:.*]] = arith.constant 0 : index
  // CHECK: %[[d0:.*]] = memref.dim %{{.*}}, %[[c0]] : memref<?x?xf32>
  // CHECK: %[[c1:.*]] = arith.constant 1 : index
  // CHECK: %[[d1:.*]] = memref.dim %{{.*}}, %[[c1]] : memref<?x?xf32>
  // CHECK: %[[mask:.*]] = vector.create_mask %[[d0]], %[[d1]] : vector<2x4xi1>
  // CHECK: vector.mask %[[mask]] {{.*}} vector.transfer_read %{{.*}} {in_bounds = [true, true]} : memref<?x?xf32>, vector<2x4xf32> } : vector<2x4xi1> -> vector<2x4xf32>
  // CHECK: vector.mask %[[mask]] {{.*}} vector.transfer_write %{{.*}} {in_bounds = [true, true]} : vector<2x4xf32>, memref<?x?xf32> } : vector<2x4xi1>
  linalg.copy ins(%A : memref<?x?xf32>) outs(%B : memref<?x?xf32>)
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [2, 4] : !transform.any_op
    transform.yield
  }
}



// -----

func.func @matmul(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  linalg.matmul ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>)
            outs(%C: memref<?x?xf32>)
  return
}

// CHECK-LABEL:   func.func @matmul(
// CHECK-SAME:      %[[A:.*]]: memref<?x?xf32>, %[[B:.*]]: memref<?x?xf32>, %[[C:.*]]: memref<?x?xf32>) {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = memref.dim %[[A]], %[[VAL_3]] : memref<?x?xf32>
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = memref.dim %[[B]], %[[VAL_5]] : memref<?x?xf32>
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_8:.*]] = memref.dim %[[A]], %[[VAL_7]] : memref<?x?xf32>
// CHECK:           %[[MASK_A:.*]] = vector.create_mask %[[VAL_4]], %[[VAL_8]] : vector<8x4xi1>
// CHECK:           %[[LOAD_A:.*]] = vector.mask %[[MASK_A]] { vector.transfer_read %[[A]]{{\[}}%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, true, true], permutation_map = #{{.*}}} : memref<?x?xf32>, vector<8x16x4xf32> } : vector<8x4xi1> -> vector<8x16x4xf32>
// CHECK:           %[[MASK_B:.*]] = vector.create_mask %[[VAL_8]], %[[VAL_6]] : vector<4x16xi1>
// CHECK:           %[[LOAD_B:.*]] = vector.mask %[[MASK_B]] { vector.transfer_read %[[B]]{{\[}}%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, true, true], permutation_map = #{{.*}}} : memref<?x?xf32>, vector<8x16x4xf32> } : vector<4x16xi1> -> vector<8x16x4xf32>
// CHECK:           %[[MASK_C:.*]] = vector.create_mask %[[VAL_4]], %[[VAL_6]] : vector<8x16xi1>
// CHECK:           %[[LOAD_C:.*]] = vector.mask %[[MASK_C]] { vector.transfer_read %[[C]]{{\[}}%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, true]} : memref<?x?xf32>, vector<8x16xf32> } : vector<8x16xi1> -> vector<8x16xf32>
// CHECK:           %[[MULF:.*]] = arith.mulf %[[LOAD_A]], %[[LOAD_B]] : vector<8x16x4xf32>
// CHECK:           %[[MASK_MULIT_RED:.*]] = vector.create_mask %[[VAL_4]], %[[VAL_6]], %[[VAL_8]] : vector<8x16x4xi1>
// CHECK:           %[[MULTI_RED:.*]] = vector.mask %[[MASK_MULIT_RED]] { vector.multi_reduction <add>, %[[MULF]], %[[LOAD_C]] [2] : vector<8x16x4xf32> to vector<8x16xf32> } : vector<8x16x4xi1> -> vector<8x16xf32>
// CHECK:           %[[C2:.*]] = arith.constant 0 : index
// CHECK:           vector.mask %[[MASK_C]] { vector.transfer_write %[[MULTI_RED]], %[[C]]{{\[}}%[[C2]], %[[C2]]] {in_bounds = [true, true]} : vector<8x16xf32>, memref<?x?xf32> } : vector<8x16xi1>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %matmul vector_sizes [8, 16, 4] : !transform.any_op
    transform.yield
  }
}


// -----

func.func @matmul_scalable(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>) {
  linalg.matmul ins(%A, %B: memref<?x?xf32>, memref<?x?xf32>)
            outs(%C: memref<?x?xf32>)
  return
}

// CHECK-LABEL:   func.func @matmul_scalable(
// CHECK-SAME:      %[[A:.*]]: memref<?x?xf32>, %[[B:.*]]: memref<?x?xf32>, %[[C:.*]]: memref<?x?xf32>) {
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = memref.dim %[[A]], %[[VAL_3]] : memref<?x?xf32>
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = memref.dim %[[B]], %[[VAL_5]] : memref<?x?xf32>
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_8:.*]] = memref.dim %[[A]], %[[VAL_7]] : memref<?x?xf32>
// CHECK:           %[[MASK_A:.*]] = vector.create_mask %[[VAL_4]], %[[VAL_8]] : vector<8x4xi1>
// CHECK:           %[[LOAD_A:.*]] = vector.mask %[[MASK_A]] { vector.transfer_read %[[A]]{{\[}}%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, true, true], permutation_map = #{{.*}}} : memref<?x?xf32>, vector<8x[16]x4xf32> } : vector<8x4xi1> -> vector<8x[16]x4xf32>
// CHECK:           %[[MASK_B:.*]] = vector.create_mask %[[VAL_8]], %[[VAL_6]] : vector<4x[16]xi1>
// CHECK:           %[[LOAD_B:.*]] = vector.mask %[[MASK_B]] { vector.transfer_read %[[B]]{{\[}}%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, true, true], permutation_map = #{{.*}}} : memref<?x?xf32>, vector<8x[16]x4xf32> } : vector<4x[16]xi1> -> vector<8x[16]x4xf32>
// CHECK:           %[[MASK_C:.*]] = vector.create_mask %[[VAL_4]], %[[VAL_6]] : vector<8x[16]xi1>
// CHECK:           %[[LOAD_C:.*]] = vector.mask %[[MASK_C]] { vector.transfer_read %[[C]]{{\[}}%{{.*}}, %{{.*}}], %{{.*}} {in_bounds = [true, true]} : memref<?x?xf32>, vector<8x[16]xf32> } : vector<8x[16]xi1> -> vector<8x[16]xf32>
// CHECK:           %[[MULF:.*]] = arith.mulf %[[LOAD_A]], %[[LOAD_B]] : vector<8x[16]x4xf32>
// CHECK:           %[[MASK_MULIT_RED:.*]] = vector.create_mask %[[VAL_4]], %[[VAL_6]], %[[VAL_8]] : vector<8x[16]x4xi1>
// CHECK:           %[[MULTI_RED:.*]] = vector.mask %[[MASK_MULIT_RED]] { vector.multi_reduction <add>, %[[MULF]], %[[LOAD_C]] [2] : vector<8x[16]x4xf32> to vector<8x[16]xf32> } : vector<8x[16]x4xi1> -> vector<8x[16]xf32>
// CHECK:           %[[C2:.*]] = arith.constant 0 : index
// CHECK:           vector.mask %[[MASK_C]] { vector.transfer_write %[[MULTI_RED]], %[[C]]{{\[}}%[[C2]], %[[C2]]] {in_bounds = [true, true]} : vector<8x[16]xf32>, memref<?x?xf32> } : vector<8x[16]xi1>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %matmul vector_sizes [8, [16], 4] : !transform.any_op
    transform.yield
  }
}
