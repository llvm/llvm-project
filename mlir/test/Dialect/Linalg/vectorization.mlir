// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

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
// CHECK:           %[[VAL_7:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_8:.*]] = vector.create_mask %[[VAL_3]], %[[VAL_5]] : vector<4x4xi1>
// CHECK:           %[[VAL_9:.*]] = vector.mask %[[VAL_8]] { vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_6]], %[[VAL_6]]], %[[VAL_7]] {in_bounds = [true, true]} : tensor<?x?xf32>, vector<4x4xf32> } : vector<4x4xi1> -> vector<4x4xf32>
// CHECK:           %[[VAL_10:.*]] = arith.constant 0.000000e+00 : f32
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
// CHECK:           %[[VAL_14:.*]] = arith.constant 0.000000e+00 : f32
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

func.func @vectorize_dynamic_reduction(%arg0: tensor<?x?xf32>,
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

// CHECK-LABEL:   @vectorize_dynamic_reduction(
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
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_8:.*]] = vector.create_mask %[[VAL_7]], %[[VAL_4]] : vector<8x32xi1>
// CHECK:           %[[VAL_9:.*]] = vector.mask %[[VAL_8]] { vector.transfer_read %[[VAL_0]][%[[VAL_5]], %[[VAL_5]]], %[[VAL_6]] {in_bounds = [true, true]} : tensor<8x?xf32>, vector<8x32xf32> } : vector<8x32xi1> -> vector<8x32xf32>
// CHECK:           %[[VAL_10:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_11:.*]] = vector.mask %[[VAL_8]] { vector.transfer_read %[[VAL_1]][%[[VAL_5]], %[[VAL_5]]], %[[VAL_10]] {in_bounds = [true, true]} : tensor<8x?xf32>, vector<8x32xf32> } : vector<8x32xi1> -> vector<8x32xf32>
// CHECK:           %[[VAL_12:.*]] = arith.constant 0.000000e+00 : f32
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
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 30 : index
// CHECK:           %[[VAL_7:.*]] = vector.create_mask %[[VAL_5]], %[[VAL_6]] : vector<8x32xi1>
// CHECK:           %[[VAL_8:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %[[VAL_0]][%[[VAL_3]], %[[VAL_3]]], %[[VAL_4]] {in_bounds = [true, true]} : tensor<8x30xf32>, vector<8x32xf32> } : vector<8x32xi1> -> vector<8x32xf32>
// CHECK:           %[[VAL_9:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_10:.*]] = vector.mask %[[VAL_7]] { vector.transfer_read %[[VAL_1]][%[[VAL_3]], %[[VAL_3]]], %[[VAL_9]] {in_bounds = [true, true]} : tensor<8x30xf32>, vector<8x32xf32> } : vector<8x32xi1> -> vector<8x32xf32>
// CHECK:           %[[VAL_11:.*]] = arith.constant 0.000000e+00 : f32
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
  //      CHECK: %[[mask_2:.*]] = vector.create_mask %[[res_d0]], %[[res_d1]] : vector<2x4xi1>
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

// Input identical as the test in vectorization-with-patterns.mlir. Output is
// different - vector sizes are inferred (rather than user-specified) and hence
// masking was used.

func.func @test_vectorize_pack(%arg0: tensor<32x8x16xf32>, %arg1: tensor<4x1x32x16x2xf32>) -> tensor<4x1x32x16x2xf32> {
  %pack = tensor.pack %arg0 outer_dims_perm = [1, 2, 0] inner_dims_pos = [2, 1] inner_tiles = [16, 2] into %arg1 : tensor<32x8x16xf32> -> tensor<4x1x32x16x2xf32>
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
    %0 = transform.structured.match ops{["tensor.pack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
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
  %pack = tensor.pack %arg0 padding_value(%pad : f32) inner_dims_pos = [2, 1] inner_tiles = [16, 2] into %arg1 : tensor<32x7x15xf32> -> tensor<32x4x1x16x2xf32>
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
    %0 = transform.structured.match ops{["tensor.pack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [32, 4, 1] : !transform.any_op
    transform.yield
  }
}

// -----

func.func @test_vectorize_dynamic_pack(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?x16x2xf32>) -> tensor<?x?x16x2xf32> {
  %pack = tensor.pack %arg0 inner_dims_pos = [1, 0] inner_tiles = [16, 2] into %arg1 : tensor<?x?xf32> -> tensor<?x?x16x2xf32>
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
//      CHECK: %[[mask_0:.*]] = vector.create_mask %[[d0]], %[[d1]], %[[c16]], %[[c2]] : vector<4x1x16x2xi1>
//      CHECK: %[[masked_write:.*]] = vector.mask %[[mask_0]] {
// CHECK-SAME:   vector.transfer_write %[[transpose]], %[[empty]][%[[c0_2]], %[[c0_2]], %[[c0_2]], %[[c0_2]]]
// CHECK-SAME:   {in_bounds = [true, true, true, true]} : vector<4x1x16x2xf32>, tensor<?x?x16x2xf32>
//      CHECK: return %[[masked_write]] : tensor<?x?x16x2xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.pack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [4, 1] : !transform.any_op
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

// -----

// CHECK-LABEL: func @test_vectorize_dynamic_shapes_unpack
func.func @test_vectorize_dynamic_shapes_unpack(%arg0: tensor<?x?xf32>, %arg1: tensor<?x?x16x2xf32>) -> tensor<?x?xf32> {
// CHECK: %[[C0:.*]] = arith.constant 0
// CHECK: %[[DIM:.*]] = tensor.dim %arg0, %[[C0]] : tensor<?x?xf32>
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[DIM0:.*]] = tensor.dim %arg0, %[[C1]] : tensor<?x?xf32>
// CHECK: %[[CST:.*]] = arith.constant 0.000000e+00
// CHECK: %[[C01:.*]] = arith.constant 0
// CHECK: %[[C02:.*]] = arith.constant 0
// CHECK: %[[DIM4:.*]] = tensor.dim %arg1, %[[C02]] : tensor<?x?x16x2xf32>
// CHECK: %[[CNST14:.*]] = arith.constant 1
// CHECK: %[[DIM6:.*]] = tensor.dim %arg1, %[[CNST14]] : tensor<?x?x16x2xf32>
// CHECK: %[[CNST16:.*]] = arith.constant 16 : index
// CHECK: %[[CNST2:.*]] = arith.constant 2 : index
// CHECK: %[[readMsk0:.*]] = vector.create_mask %[[DIM4]], %[[DIM6]], %[[CNST16]], %[[CNST2]] : vector<2x1x16x2xi1>
// CHECK: %[[read0:.*]] = vector.mask %[[readMsk0]] {{.*}} vector.transfer_read %{{.*}} : tensor<?x?x16x2xf32>, vector<2x1x16x2xf32> } : vector<2x1x16x2xi1> -> vector<2x1x16x2xf32>
// CHECK: %[[trans0:.*]] = vector.transpose %[[read0]], [0, 3, 1, 2] : vector<2x1x16x2xf32> to vector<2x2x1x16xf32>
// CHECK: %[[sc0:.*]] = vector.shape_cast %[[trans0]] : vector<2x2x1x16xf32> to vector<4x16xf32>
// CHECK: %[[empt0:.*]] = tensor.empty
// CHECK: %[[writeMsk0:.*]] = vector.create_mask {{.*}} : vector<4x16xi1>
// CHECK: %[[write0:.*]] = vector.mask %[[writeMsk0:.*]] {{.*}} vector.transfer_write %[[sc0]], %[[empt0]]
// CHECK: return %[[write0]]
 %ret = tensor.unpack %arg1 inner_dims_pos = [1, 0] inner_tiles = [16, 2] into %arg0 : tensor<?x?x16x2xf32> -> tensor<?x?xf32>
 return %ret : tensor<?x?xf32>
}
module attributes {transform.with_named_sequence} {
 transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
   %0 = transform.structured.match ops{["tensor.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
   transform.structured.vectorize %0 vector_sizes [4, 16] : !transform.any_op
   transform.yield
 }
}

// -----

// CHECK-LABEL: func @test_vectorize_unpack
func.func @test_vectorize_unpack(%source: tensor<8x8x32x16xf32>, %dest: tensor<256x128xf32>) -> tensor<256x128xf32> {
    // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[C0:.*]]= arith.constant 0 : index
    // CHECK: %[[C8:.*]] = arith.constant 8 : index
    // CHECK: %[[C80:.*]] = arith.constant 8 : index
    // CHECK: %[[C32:.*]] = arith.constant 32 : index
    // CHECK: %[[C16:.*]] = arith.constant 16 : index
    // CHECK: %[[MSK0:.*]] = vector.create_mask %[[C8]], %[[C80]], %[[C32]], %[[C16]] : vector<16x8x32x16xi1>
    // CHECK: %[[READ0:.*]] = vector.mask %[[MSK0]] {{.*}} : vector<16x8x32x16xi1> -> vector<16x8x32x16xf32>
    // CHECK: %[[TRANSP0:.*]] = vector.transpose %[[READ0]], [0, 2, 1, 3] : vector<16x8x32x16xf32> to vector<16x32x8x16xf32>
    // CHECK: %[[SHAPC:.*]] = vector.shape_cast %[[TRANSP0]] : vector<16x32x8x16xf32> to vector<512x128xf32>
    // CHECK: %[[EMPT:.*]] = tensor.empty() : tensor<256x128xf32>
    // CHECK: %[[C01:.*]] = arith.constant 0 : index
    // CHECK: %[[C256:.*]] = arith.constant 256 : index
    // CHECK: %[[C128:.*]] = arith.constant 128 : index
    // CHECK: %[[WRITEMSK:.*]] = vector.create_mask %[[C256]], %[[C128]] : vector<512x128xi1>
    // CHECK: %[[WRIT:.*]] = vector.mask %[[WRITEMSK]] {{.*}} : vector<512x128xi1> -> tensor<256x128xf32>
    // CHECK: return %[[WRIT]] : tensor<256x128xf32>
   %0 = tensor.unpack %source inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %dest : tensor<8x8x32x16xf32> -> tensor<256x128xf32>
   return %0 : tensor<256x128xf32>
 }
 module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
   transform.structured.vectorize %0 vector_sizes [512, 128] : !transform.any_op
    transform.yield
  }
}

// -----

// CHECK-LABEL: func @test_vectorize_unpack_no_masks
func.func @test_vectorize_unpack_no_masks(%source: tensor<8x8x32x16xf32>, %dest: tensor<256x128xf32>) -> tensor<256x128xf32> {
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[READ:.*]] = vector.transfer_read {{.*}} : tensor<8x8x32x16xf32>, vector<8x8x32x16xf32>
  // CHECK: %[[TRANSP:.*]] = vector.transpose %[[READ]], [0, 2, 1, 3] : vector<8x8x32x16xf32> to vector<8x32x8x16xf32>
  // CHECK: %[[SHAPC:.*]] = vector.shape_cast %[[TRANSP]] : vector<8x32x8x16xf32> to vector<256x128xf32>
  // CHECK: %[[EMPT:.*]] = tensor.empty() : tensor<256x128xf32>
  // CHECK: %[[C00:.*]] = arith.constant 0 : index
  // CHECK: %[[WRIT:.*]] = vector.transfer_write %[[SHAPC]], {{.*}} : vector<256x128xf32>, tensor<256x128xf32>
  // CHECK: return %[[WRIT]] : tensor<256x128xf32>
   %0 = tensor.unpack %source inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %dest : tensor<8x8x32x16xf32> -> tensor<256x128xf32>
   return %0 : tensor<256x128xf32>
 }
 module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
   transform.structured.vectorize %0 vector_sizes [256, 128] : !transform.any_op
    transform.yield
  }
 }

  // -----

  // CHECK-LABEL: test_vectorize_unpack_with_outer_perm
  func.func @test_vectorize_unpack_with_outer_perm(%source: tensor<8x8x32x16xf32>, %dest: tensor<256x128xf32>) -> tensor<256x128xf32> {
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[READ:.*]] = vector.transfer_read {{.*}} : tensor<8x8x32x16xf32>, vector<8x8x32x16xf32>
  // CHECK: %[[TRANSP:.*]] = vector.transpose %[[READ]], [1, 2, 0, 3] : vector<8x8x32x16xf32> to vector<8x32x8x16xf32>
  // CHECK: %[[SHAPC:.*]] = vector.shape_cast %[[TRANSP]] : vector<8x32x8x16xf32> to vector<256x128xf32>
  // CHECK: %[[EMPT:.*]] = tensor.empty() : tensor<256x128xf32>
  // CHECK: %[[C00:.*]] = arith.constant 0 : index
  // CHECK: %[[WRIT:.*]] = vector.transfer_write %[[SHAPC]], {{.*}} : vector<256x128xf32>, tensor<256x128xf32>
  // CHECK: return %[[WRIT]] : tensor<256x128xf32>
   %0 = tensor.unpack %source outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %dest : tensor<8x8x32x16xf32> -> tensor<256x128xf32>
   return %0 : tensor<256x128xf32>
 }
 module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
   transform.structured.vectorize %0 vector_sizes [256, 128] : !transform.any_op
    transform.yield
  }
}

  // -----

// CHECK-LABEL: test_vectorize_pack_no_vector_sizes
func.func @test_vectorize_pack_no_vector_sizes(%arg0: tensor<64x4xf32>, %arg1: tensor<2x4x16x2xf32>) -> tensor<2x4x16x2xf32> {
  %pack = tensor.pack %arg0 outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [16, 2] into %arg1 : tensor<64x4xf32> -> tensor<2x4x16x2xf32>
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
    %0 = transform.structured.match ops{["tensor.pack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 : !transform.any_op
    transform.yield
  }
}

  // -----

// CHECK-LABEL: test_vectorize_padded_pack_no_vector_sizes
func.func @test_vectorize_padded_pack_no_vector_sizes(%arg0: tensor<32x7x15xf32>, %arg1: tensor<32x4x1x16x2xf32>) -> tensor<32x4x1x16x2xf32> {
  %pad = arith.constant 0.000000e+00 : f32
  %pack = tensor.pack %arg0 padding_value(%pad : f32) inner_dims_pos = [2, 1] inner_tiles = [16, 2] into %arg1 : tensor<32x7x15xf32> -> tensor<32x4x1x16x2xf32>
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
    %0 = transform.structured.match ops{["tensor.pack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 : !transform.any_op
    transform.yield
  }
}

  // -----

func.func @test_vectorize_unpack_no_vector_sizes(%source: tensor<8x8x32x16xf32>, %dest: tensor<256x128xf32>) -> tensor<256x128xf32> {
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[READ:.*]] = vector.transfer_read {{.*}} : tensor<8x8x32x16xf32>, vector<8x8x32x16xf32>
  // CHECK: %[[TRANSP:.*]] = vector.transpose %[[READ]], [0, 2, 1, 3] : vector<8x8x32x16xf32> to vector<8x32x8x16xf32>
  // CHECK: %[[SHAPC:.*]] = vector.shape_cast %[[TRANSP]] : vector<8x32x8x16xf32> to vector<256x128xf32>
  // CHECK: %[[EMPT:.*]] = tensor.empty() : tensor<256x128xf32>
  // CHECK: %[[C00:.*]] = arith.constant 0 : index
  // CHECK: %[[WRIT:.*]] = vector.transfer_write %[[SHAPC]], {{.*}} : vector<256x128xf32>, tensor<256x128xf32>
  // CHECK: return %[[WRIT]] : tensor<256x128xf32>
   %0 = tensor.unpack %source inner_dims_pos = [0, 1] inner_tiles = [32, 16] into %dest : tensor<8x8x32x16xf32> -> tensor<256x128xf32>
   return %0 : tensor<256x128xf32>
 }
 module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
   transform.structured.vectorize %0 : !transform.any_op
    transform.yield
  }
 }

  // -----

func.func @test_vectorize_unpack_no_vector_sizes_slice_output(%source: tensor<8x4x16x16xf32>, %dest: tensor<64x127xf32>) -> tensor<64x127xf32> {
  //      CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  //      CHECK: %[[C0:.*]] = arith.constant 0 : index
  //      CHECK: %[[READ:.*]] = vector.transfer_read {{.*}} : tensor<8x4x16x16xf32>, vector<8x4x16x16xf32>
  //      CHECK: %[[TRANSP:.*]] = vector.transpose %[[READ]], [1, 2, 0, 3] : vector<8x4x16x16xf32> to vector<4x16x8x16xf32>
  //      CHECK: %[[SHAPC:.*]] = vector.shape_cast %[[TRANSP]] : vector<4x16x8x16xf32> to vector<64x128xf32>
  //      CHECK: %[[EMPT:.*]] = tensor.empty() : tensor<64x127xf32>
  //      CHECK: %[[C00:.*]] = arith.constant 0 : index
  //      CHECK: %[[WRIT:.*]] = vector.transfer_write %[[SHAPC]], %[[EMPT]]{{\[}}%[[C00]], %[[C00]]]
  // CHECK-SAME:  {in_bounds = [true, false]} : vector<64x128xf32>, tensor<64x127xf32>
  //      CHECK: return %[[WRIT]] : tensor<64x127xf32>
   %0 = tensor.unpack %source outer_dims_perm = [1, 0] inner_dims_pos = [0, 1] inner_tiles = [16, 16] into %dest : tensor<8x4x16x16xf32> -> tensor<64x127xf32>
   return %0 : tensor<64x127xf32>
 }
 module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
   transform.structured.vectorize %0 : !transform.any_op
    transform.yield
  }
 }

// -----

func.func @test_vectorize_unpack_no_vector_sizes_permute(%source: tensor<4x7x4xf32>, %dest: tensor<7x16xf32>) -> tensor<7x16xf32> {
   %0 = tensor.unpack %source outer_dims_perm=[1, 0] inner_dims_pos = [1] inner_tiles = [4] into %dest : tensor<4x7x4xf32> -> tensor<7x16xf32>
   return %0 : tensor<7x16xf32>
 }
  // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK: %[[C0:.*]] = arith.constant 0 : index
  // CHECK: %[[READ:.*]] = vector.transfer_read {{.*}} : tensor<4x7x4xf32>, vector<4x7x4xf32>
  // CHECK: %[[TRANSP:.*]] = vector.transpose %[[READ]], [1, 0, 2] : vector<4x7x4xf32> to vector<7x4x4xf32>
  // CHECK: %[[SHAPC:.*]] = vector.shape_cast %[[TRANSP]] : vector<7x4x4xf32> to vector<7x16xf32>
  // CHECK: %[[EMPT:.*]] = tensor.empty() : tensor<7x16xf32>
  // CHECK: %[[C00:.*]] = arith.constant 0 : index
  // CHECK: %[[WRIT:.*]] = vector.transfer_write %[[SHAPC]], {{.*}} : vector<7x16xf32>, tensor<7x16xf32>
  // CHECK: return %[[WRIT]] : tensor<7x16xf32>
 module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.unpack"]} in %arg0 : (!transform.any_op) -> !transform.any_op
   transform.structured.vectorize %0 : !transform.any_op
    transform.yield
  }
 }

// -----

///----------------------------------------------------------------------------------------
/// tensor.insert_slice
///----------------------------------------------------------------------------------------

func.func private @insert_slice_static_sizes(%source: tensor<?x3x?x1xi32>) -> tensor<5x3xi32> {
  %c2 = arith.constant 2 : index
  %init = tensor.empty() : tensor<5x3xi32>

  %source_slice = tensor.extract_slice %source[0, %c2, 0, 0] [1, 1, 5, 1] [1, 1, 1, 1] : tensor<?x3x?x1xi32> to tensor<5x1xi32>
  %res = tensor.insert_slice %source_slice into %init[0, %c2] [5, 1] [1, 1] : tensor<5x1xi32> into tensor<5x3xi32>

  return %res : tensor<5x3xi32>
}

// CHECK-LABEL:   func.func private @insert_slice_static_sizes(
// CHECK-SAME:      %[[SEC:.*]]: tensor<?x3x?x1xi32>) -> tensor<5x3xi32> {
// CHECK:           %[[C_2:.*]] = arith.constant 2 : index
// CHECK:           %[[INIT:.*]] = tensor.empty() : tensor<5x3xi32>
// CHECK:           %[[SRC_SLICE:.*]] = tensor.extract_slice %[[SEC]][0, %[[C_2]], 0, 0] [1, 1, 5, 1] [1, 1, 1, 1] : tensor<?x3x?x1xi32> to tensor<5x1xi32>
// CHECK:           %[[PAD:.*]] = arith.constant 0 : i32
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[C_5:.*]] = arith.constant 5 : index
// CHECK:           %[[C_1:.*]] = arith.constant 1 : index
// CHECK:           %[[MASK:.*]] = vector.create_mask %[[C_5]], %[[C_1]] : vector<8x1xi1>
// CHECK:           %[[READ:.*]] = vector.mask %[[MASK]] { vector.transfer_read %[[SRC_SLICE]][%[[C0]], %[[C0]]], %[[PAD]] : tensor<5x1xi32>, vector<8x1xi32> } : vector<8x1xi1> -> vector<8x1xi32>
// CHECK:           %[[C_0:.*]] = arith.constant 0 : index
// CHECK:           %[[RES:.*]] = vector.transfer_write %[[READ]], %[[INIT]][%[[C_0]], %[[C_2]]] : vector<8x1xi32>, tensor<5x3xi32>
// CHECK:           return %[[RES]] : tensor<5x3xi32>

 module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.insert_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [8, 1] : !transform.any_op
    transform.yield
  }
 }

// -----

// One of the _source_ dimensions is dynamic (but _destination_ dimensions are static).

func.func private @insert_slice_dynamic_src_dim(%source: tensor<?x3x?x1xi32>, %size: index) -> tensor<5x3xi32> {
  %c2 = arith.constant 2 : index
  %init = tensor.empty() : tensor<5x3xi32>

  %source_slice = tensor.extract_slice %source[0, %c2, 0, 0] [1, 1, %size, 1] [1, 1, 1, 1] : tensor<?x3x?x1xi32> to tensor<?x1xi32>
  %res = tensor.insert_slice %source_slice into %init[0, %c2] [%size, 1] [1, 1] : tensor<?x1xi32> into tensor<5x3xi32>

  return %res : tensor<5x3xi32>
}

// CHECK-LABEL:   func.func private @insert_slice_dynamic_src_dim(
// CHECK-SAME:      %[[SRC:.*]]: tensor<?x3x?x1xi32>,
// CHECK-SAME:      %[[SIZE:.*]]: index) -> tensor<5x3xi32> {
// CHECK:           %[[C_2:.*]] = arith.constant 2 : index
// CHECK:           %[[INIT:.*]] = tensor.empty() : tensor<5x3xi32>
// CHECK:           %[[SRC_SLICE:.*]] = tensor.extract_slice %[[SRC]][0, %[[C_2]], 0, 0] [1, 1, %[[SIZE]], 1] [1, 1, 1, 1] : tensor<?x3x?x1xi32> to tensor<?x1xi32>
// CHECK-DAG:       %[[PAD:.*]] = arith.constant 0 : i32
// CHECK-DAG:       %[[C_1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C_0:.*]] = arith.constant 0 : index
// CHECK:           %[[MASK:.*]] = vector.create_mask %[[SIZE]], %[[C_1]] : vector<8x1xi1>
// CHECK:           %[[READ:.*]] = vector.mask %[[MASK]] { vector.transfer_read %[[SRC_SLICE]][%[[C_0]], %[[C_0]]], %[[PAD]] : tensor<?x1xi32>, vector<8x1xi32> } : vector<8x1xi1> -> vector<8x1xi32>
// CHECK:           %[[C_0_1:.*]] = arith.constant 0 : index
// CHECK:           %[[RES:.*]] = vector.transfer_write %[[READ]], %[[INIT]][%[[C_0_1]], %[[C_2]]] : vector<8x1xi32>, tensor<5x3xi32>
// CHECK:           return %[[RES]] : tensor<5x3xi32>

 module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["tensor.insert_slice"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 vector_sizes [8, 1] : !transform.any_op
    transform.yield
  }
 }
