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
  //  CHECK-DAG: %[[empty:.*]] = tensor.empty() : tensor<2x4xf32>
  //      CHECK: %[[d0:.*]] = tensor.dim {{.*}} : tensor<?x?xf32>
  //      CHECK: %[[d1:.*]] = tensor.dim {{.*}} : tensor<?x?xf32>
  //      CHECK: %[[mask:.*]] = vector.create_mask %[[d0]], %[[d1]] : vector<2x4xi1>
  //  CHECK-DAG: %[[c0_2:.*]] = arith.constant 0 : index
  //      CHECK: %[[masked_read:.*]] = vector.mask %[[mask]] {
  // CHECK-SAME:   vector.transfer_read %{{.*}}[%[[c0_2]], %[[c0_2]]], %[[c42]]
  // CHECK-SAME:   {in_bounds = [true, true]} : tensor<?x?xf32>, vector<2x4xf32>
  // CHECK-SAME: } : vector<2x4xi1> -> vector<2x4xf32>
  //      CHECK: vector.transfer_write %[[masked_read]], %[[empty]][%[[c0_2]], %[[c0_2]]]
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
  //  CHECK-DAG: %[[empty:.*]] = tensor.empty(%[[res_d0]], %[[res_d1]]) : tensor<?x?xf32>
  //      CHECK: %[[d0:.*]] = tensor.dim {{.*}} : tensor<?x?xf32>
  //      CHECK: %[[d1:.*]] = tensor.dim {{.*}} : tensor<?x?xf32>
  //      CHECK: %[[mask:.*]] = vector.create_mask %[[d0]], %[[d1]] : vector<2x4xi1>
  //  CHECK-DAG: %[[c0_2:.*]] = arith.constant 0 : index
  //      CHECK: %[[masked_read:.*]] = vector.mask %[[mask]] {
  // CHECK-SAME:   vector.transfer_read %{{.*}}[%[[c0_2]], %[[c0_2]]], %[[c42]]
  // CHECK-SAME:   {in_bounds = [true, true]} : tensor<?x?xf32>, vector<2x4xf32>
  // CHECK-SAME: } : vector<2x4xi1> -> vector<2x4xf32>
  //      CHECK: %[[mask_2:.*]] = vector.create_mask %[[res_d0]], %[[res_d1]] : vector<2x4xi1>
  //      CHECK: %[[masked_write:.*]] = vector.mask %[[mask_2]] {
  // CHECK-SAME: vector.transfer_write %[[masked_read]], %[[empty]][%[[c0_2]], %[[c0_2]]]
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
