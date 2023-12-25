// RUN: mlir-opt %s -transform-interpreter -split-input-file | FileCheck %s

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @vectorize_1d_tensor_extract(%arg0: tensor<3xf32>, %arg1: tensor<4x3xi32>, %arg2: tensor<4x7x3x2xf32>) -> tensor<4x7x3x2xf32> {
  %1 = linalg.generic {
    indexing_maps = [#map0, #map1],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%arg1 : tensor<4x3xi32>) outs(%arg2 : tensor<4x7x3x2xf32>) {
  ^bb0(%arg3: i32, %arg4: f32):
    %2 = arith.index_cast %arg3 : i32 to index
    %3 = tensor.extract %arg0[%2] : tensor<3xf32>
    linalg.yield %3 : f32
  } -> tensor<4x7x3x2xf32>
  return %1 : tensor<4x7x3x2xf32>
}
// CHECK-LABEL: func.func @vectorize_1d_tensor_extract
// CHECK-SAME:    %[[ARG0:.*]]: tensor<3xf32>
// CHECK-SAME:    %[[ARG1:.*]]: tensor<4x3xi32>
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[MASK:.*]] = arith.constant dense<true> : vector<4x7x3x2xi1>
// CHECK: %[[PASSTHRU:.*]] = arith.constant dense<0.000000e+00> : vector<4x7x3x2xf32>
// CHECK: %[[V0:.*]] = vector.transfer_read %[[ARG1]]
// CHECK: %[[CAST:.*]] = arith.index_cast %[[V0]]
// CHECK: %[[BROADCAST:.*]] = vector.broadcast %[[CAST]]
// CHECK: %[[INDICES:.*]] = vector.transpose %[[BROADCAST]]
// CHECK: %[[GATHER:.*]] = vector.gather %[[ARG0]][%[[C0]]] [%[[INDICES]]], %[[MASK]], %[[PASSTHRU]]
// CHECK: vector.transfer_write %[[GATHER]]

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----
#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @vectorize_nd_tensor_extract_constant_idx(%arg0: tensor<3x3xf32>, %arg2: tensor<1x1x3xf32>) -> tensor<1x1x3xf32> {
  %c0 = arith.constant 1 : index
  %c1 = arith.constant 2 : index
  %2 = linalg.generic {
    indexing_maps = [#map],
    iterator_types = ["parallel", "parallel", "parallel"]
  } outs(%arg2 : tensor<1x1x3xf32>) {
  ^bb0(%arg4: f32):
    %7 = tensor.extract %arg0[%c0, %c1] : tensor<3x3xf32>
    linalg.yield %7 : f32
  } -> tensor<1x1x3xf32>
  return %2 : tensor<1x1x3xf32>
}

// CHECK: #[[$MAP:.*]] = affine_map<(d0, d1) -> (0, 0, 0)>
// CHECK-LABEL:   func.func @vectorize_nd_tensor_extract_constant_idx(
// CHECK-SAME:      %[[ARG_0:.*]]: tensor<3x3xf32>,
// CHECK-SAME:      %[[ARG_1:.*]]: tensor<1x1x3xf32>) -> tensor<1x1x3xf32> {
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:       arith.constant 0.000000e+00 : f32
// CHECK-DAG:       %[[C0_f32:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[READ:.*]] = vector.transfer_read  %[[ARG_0]][%[[C1]], %[[C2]]], %[[C0_f32]] {in_bounds = [true, true, true], permutation_map = #[[$MAP]]} : tensor<3x3xf32>, vector<1x1x3xf32>
// CHECK:           %[[C0_4:.*]] = arith.constant 0 : index
// CHECK:           vector.transfer_write %[[READ]], %[[ARG_1]][%[[C0_4]], %[[C0_4]], %[[C0_4]]]  : vector<1x1x3xf32>, tensor<1x1x3xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %0 { vectorize_nd_extract }  : !transform.any_op
    transform.yield
   }
}

// -----

#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @vectorize_nd_tensor_extract_transfer_read_basic(%arg0: tensor<3x3x3xf32>, %arg2: tensor<1x1x3xf32>) -> tensor<1x1x3xf32> {
  %1 = linalg.generic {
    indexing_maps = [#map1],
    iterator_types = ["parallel", "parallel", "parallel"]
  } outs(%arg2 : tensor<1x1x3xf32>) {
  ^bb0(%arg4: f32):
    %2 = linalg.index 0 : index
    %3 = linalg.index 1 : index
    %4 = linalg.index 2 : index
    %5 = tensor.extract %arg0[%2, %3, %4] : tensor<3x3x3xf32>
    linalg.yield %5 : f32
  } -> tensor<1x1x3xf32>
  return %1 : tensor<1x1x3xf32>
}

// CHECK-LABEL: func.func @vectorize_nd_tensor_extract_transfer_read_basic
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x3x3xf32>
// CHECK-SAME: %[[ARG1:.*]]: tensor<1x1x3xf32>) -> tensor<1x1x3xf32> {
//      CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<1xindex>
//      CHECK: %[[CST_0:.*]] = arith.constant dense<[0, 1, 2]> : vector<3xindex>
//      CHECK: %[[CST_1:.*]] = arith.constant 0.000000e+00 : f32
//      CHECK: %[[C0:.*]] = arith.constant 0 : index
//      CHECK: %[[E0:.*]] = vector.extract %[[CST]][0] : index from vector<1xindex>
//      CHECK: %[[E1:.*]] = vector.extract %[[CST]][0] : index from vector<1xindex>
//      CHECK: %[[E2:.*]] = vector.extract %[[CST_0]][0] : index from vector<3xindex>
//      CHECK: %[[R1:.*]] = vector.transfer_read %[[ARG0]][%[[E0]], %[[E1]], %[[E2]]], %[[CST_1]] {in_bounds = [true, true, true]} : tensor<3x3x3xf32>, vector<1x1x3xf32>
//      CHECK: %[[RES:.*]] = vector.transfer_write %[[R1]], %[[ARG1]][%[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true, true, true]} : vector<1x1x3xf32>, tensor<1x1x3xf32>
//      CHECK: return %[[RES]] : tensor<1x1x3xf32>
//      CHECK: }


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_nd_extract } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

func.func @vectorize_nd_tensor_extract_transfer_read_complex(%6: tensor<45x80x16xf32>, %arg0: index, %arg2: index, %arg1: index, %arg4: index, %extracted_slice : tensor<1x4xf32>) -> tensor<1x4xf32> {
  %c79 = arith.constant 79 : index
  %25 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } outs(%extracted_slice : tensor<1x4xf32>) {
  ^bb0(%out: f32):
    %26 = linalg.index 0 : index
    %27 = arith.addi %arg0, %26 : index
    %28 = arith.addi %27, %arg2 : index
    %29 = linalg.index 1 : index
    %30 = arith.addi %arg1, %29 : index
    %31 = arith.addi %30, %arg4 : index
    %extracted = tensor.extract %6[%28, %c79, %31] : tensor<45x80x16xf32>
    linalg.yield %extracted : f32
  } -> tensor<1x4xf32>
  return %25 : tensor<1x4xf32>
}

/// CHECK-LABEL:   func.func @vectorize_nd_tensor_extract_transfer_read_complex(
// CHECK-SAME:      %[[VAL_0:.*]]: tensor<45x80x16xf32>,
// CHECK-SAME:      %[[VAL_1:.*]]: index, %[[VAL_2:.*]]: index, %[[VAL_3:.*]]: index, %[[VAL_4:.*]]: index,
// CHECK-SAME:      %[[VAL_5:.*]]: tensor<1x4xf32>) -> tensor<1x4xf32> {
// CHECK:           %[[VAL_6:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
// CHECK:           %[[VAL_8:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_9:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_10:.*]] = arith.constant 79 : index
// CHECK:           %[[VAL_11:.*]] = arith.addi %[[VAL_1]], %[[VAL_2]] : index
// CHECK:           %[[VAL_13:.*]] = vector.broadcast %[[VAL_3]] : index to vector<4xindex>
// CHECK:           %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_6]] : vector<4xindex>
// CHECK:           %[[VAL_15:.*]] = vector.broadcast %[[VAL_4]] : index to vector<4xindex>
// CHECK:           %[[VAL_16:.*]] = arith.addi %[[VAL_14]], %[[VAL_15]] : vector<4xindex>
// CHECK:           %[[VAL_18:.*]] = vector.extract %[[VAL_16]][0] : index from vector<4xindex>
// CHECK:           %[[VAL_20:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_11]], %[[VAL_10]], %[[VAL_18]]], %[[VAL_8]] {in_bounds = [true, true]} : tensor<45x80x16xf32>, vector<1x4xf32>
// CHECK:           %[[VAL_21:.*]] = vector.transfer_write %[[VAL_20]], %[[VAL_5]]{{\[}}%[[VAL_9]], %[[VAL_9]]] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<1x4xf32>
// CHECK:           return %[[VAL_21]] : tensor<1x4xf32>
// CHECK:         }

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
     %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_nd_extract } : (!transform.any_op) -> !transform.any_op
     transform.yield
   }
}

// -----

#map0 = affine_map<(d0, d1, d2, d3) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
func.func @vectorize_nd_tensor_extract_index_from_tensor(%arg0: tensor<3x3xf32>, %arg1: tensor<4x3xi32>, %arg2: tensor<4x3xi32>, %arg3: tensor<4x7x2xf32>, %arg4: tensor<4x7x3x2xf32>) -> tensor<4x7x3x2xf32> {
  %2 = linalg.generic {
    indexing_maps = [#map0, #map0, #map1, #map2],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%arg1, %arg2, %arg3 : tensor<4x3xi32>, tensor<4x3xi32>, tensor<4x7x2xf32>) outs(%arg4 : tensor<4x7x3x2xf32>) {
  ^bb0(%arg5: i32, %arg6: i32, %arg7: f32, %arg8: f32):
    %3 = arith.index_cast %arg5 : i32 to index
    %4 = arith.index_cast %arg6 : i32 to index
    %7 = tensor.extract %arg0[%3, %4] : tensor<3x3xf32>
    linalg.yield %7 : f32
  } -> tensor<4x7x3x2xf32>
  return %2 : tensor<4x7x3x2xf32>
}
// CHECK-LABEL: func.func @vectorize_nd_tensor_extract_index_from_tensor
// CHECK-SAME: %[[ARG0:.*]]: tensor<3x3xf32>
// CHECK-SAME: %[[ARG1:arg1]]: tensor<4x3xi32>
// CHECK-SAME: %[[ARG2:arg2]]: tensor<4x3xi32>
// CHECK-SAME: %[[ARG3:.*]]: tensor<4x7x2xf32>
// CHECK-SAME: %[[ARG4:.*]]: tensor<4x7x3x2xf32>
// CHECK:    %[[C0:.*]] = arith.constant 0 : index
// CHECK:    %[[C0_i32:.*]] = arith.constant 0 : i32
// CHECK:    %[[CST:.*]] = arith.constant dense<3> : vector<7x2x4x3xindex>
// CHECK:    %[[CST_1:.*]] = arith.constant dense<true> : vector<4x7x3x2xi1>
// CHECK:    %[[PASSTHRU:.*]] = arith.constant dense<0.000000e+00> : vector<4x7x3x2xf32>
// CHECK:    %[[V0:.*]] = vector.transfer_read %[[ARG1]][%[[C0]], %[[C0]]], %[[C0_i32]] {in_bounds = [true, true]} : tensor<4x3xi32>, vector<4x3xi32>
// CHECK:    %[[V1:.*]] = vector.transfer_read %[[ARG2]][%[[C0]], %[[C0]]], %[[C0_i32]] {in_bounds = [true, true]} : tensor<4x3xi32>, vector<4x3xi32>
// CHECK:    %[[CAST:.*]] = arith.index_cast %[[V0]] : vector<4x3xi32> to vector<4x3xindex>
// CHECK:    %[[B1:.*]] = vector.broadcast %[[CAST]] : vector<4x3xindex> to vector<7x2x4x3xindex>
// CHECK:    %[[CAST_1:.*]] = arith.index_cast %[[V1]] : vector<4x3xi32> to vector<4x3xindex>
// CHECK:    %[[B2:.*]] = vector.broadcast %[[CAST_1]] : vector<4x3xindex> to vector<7x2x4x3xindex>
// CHECK:    %[[MULI:.*]] = arith.muli %[[B1]], %[[CST]] : vector<7x2x4x3xindex>
// CHECK:    %[[ADDI:.*]] = arith.addi %[[B2]], %[[MULI]] : vector<7x2x4x3xindex>
// CHECK:    %[[T:.*]] = vector.transpose %[[ADDI]], [2, 0, 3, 1] : vector<7x2x4x3xindex> to vector<4x7x3x2xindex>
// CHECK:    %[[GATHER:.*]] = vector.gather %[[ARG0]][%[[C0]], %[[C0]]] [%[[T]]], %[[CST_1]], %[[PASSTHRU]] : tensor<3x3xf32>, vector<4x7x3x2xindex>, vector<4x7x3x2xi1>, vector<4x7x3x2xf32> into vector<4x7x3x2xf32>
// CHECK:    vector.transfer_write %[[GATHER]], %[[ARG4]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true, true, true, true]} : vector<4x7x3x2xf32>, tensor<4x7x3x2xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
    %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_nd_extract } : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
// -----

#map = affine_map<(d0) -> (d0)>
func.func @vectorize_nd_tensor_extract_contiguous_and_gather(%arg0: tensor<6xf32>, %arg1: tensor<5xi32>) -> tensor<5xf32> {
 %c5 = arith.constant 5 : index
 %c0 = arith.constant 0 : index
 %0 = tensor.empty() : tensor<5xf32>
 %1 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel"]} outs(%0 : tensor<5xf32>) {
 ^bb0(%out: f32):
   %2 = linalg.index 0 : index
   %extracted = tensor.extract %arg1[%2] : tensor<5xi32>
   %3 = arith.index_cast %extracted : i32 to index
   %4 = arith.maxsi %3, %c0 : index
   %5 = arith.minsi %4, %c5 : index
   %extracted_0 = tensor.extract %arg0[%5] : tensor<6xf32>
   linalg.yield %extracted_0 : f32
 } -> tensor<5xf32>
 return %1 : tensor<5xf32>
}

// CHECK-LABEL:   func.func @vectorize_nd_tensor_extract_contiguous_and_gather(
// CHECK-SAME:                    %[[VAL_0:.*]]: tensor<6xf32>
// CHECK-SAME:                    %[[VAL_1:.*]]: tensor<5xi32>
// CHECK:           %[[CST:.*]] = arith.constant dense<[0, 1, 2, 3, 4]> : vector<5xindex>
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<0> : vector<5xindex>
// CHECK:           %[[VAL_5:.*]] = arith.constant dense<5> : vector<5xindex>
// CHECK:           %[[VAL_6:.*]] = arith.constant dense<true> : vector<5xi1>
// CHECK:           %[[VAL_7:.*]] = arith.constant dense<0.000000e+00> : vector<5xf32>
// CHECK:           %[[C0:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_8:.*]] = tensor.empty() : tensor<5xf32>
// CHECK:           %[[E0:.*]] = vector.extract %[[CST]][0] : index from vector<5xindex>
// CHECK:           %[[VAL_9:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[E0]]], %[[VAL_3]] {in_bounds = [true]} : tensor<5xi32>, vector<5xi32>
// CHECK:           %[[VAL_10:.*]] = arith.index_cast %[[VAL_9]] : vector<5xi32> to vector<5xindex>
// CHECK:           %[[VAL_11:.*]] = arith.maxsi %[[VAL_10]], %[[VAL_4]] : vector<5xindex>
// CHECK:           %[[VAL_12:.*]] = arith.minsi %[[VAL_11]], %[[VAL_5]] : vector<5xindex>
// CHECK:           %[[VAL_13:.*]] = vector.gather %[[VAL_0]]{{\[}}%[[C0]]] {{\[}}%[[VAL_12]]], %[[VAL_6]], %[[VAL_7]] : tensor<6xf32>, vector<5xindex>, vector<5xi1>, vector<5xf32> into vector<5xf32>
// CHECK:           %[[VAL_14:.*]] = vector.transfer_write %[[VAL_13]], %[[VAL_8]]{{\[}}%[[C0]]] {in_bounds = [true]} : vector<5xf32>, tensor<5xf32>
// CHECK:           return %[[VAL_14]] : tensor<5xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
     %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_nd_extract } : (!transform.any_op) -> !transform.any_op
     transform.yield
   }
}

// -----

// The vectorizer converts `affine.apply` so that the subsequent Ops can be vectorised based on the converted ops. Contiguous load.
func.func @vectorize_nd_tensor_extract_with_affine_apply_contiguous(%6: tensor<80x16xf32>, %arg0: index, %extracted_slice : tensor<1x4xf32>) -> tensor<1x4xf32> {
  %c79 = arith.constant 79 : index
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } outs(%extracted_slice : tensor<1x4xf32>) {
  ^bb0(%out: f32):
    %2 = linalg.index 1 : index
    %3 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%2, %arg0)
    %extracted = tensor.extract %6[%c79, %3] : tensor<80x16xf32>
    linalg.yield %extracted : f32
  } -> tensor<1x4xf32>
  return %1 : tensor<1x4xf32>
}

// CHECK-LABEL:   func.func @vectorize_nd_tensor_extract_with_affine_apply_contiguous(
// CHECK-SAME:                                                                        %[[VAL_0:.*]]: tensor<80x16xf32>,
// CHECK-SAME:                                                                        %[[VAL_1:.*]]: index,
// CHECK-SAME:                                                                        %[[VAL_2:.*]]: tensor<1x4xf32>) -> tensor<1x4xf32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
// CHECK:           %[[VAL_5:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 79 : index
// CHECK:           %[[VAL_8:.*]] = vector.broadcast %[[VAL_1]] : index to vector<4xindex>
// CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_3]] : vector<4xindex>
// CHECK:           %[[VAL_10:.*]] = vector.extract %[[VAL_9]][0] : index from vector<4xindex>
// CHECK:           %[[VAL_11:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_7]], %[[VAL_10]]], %[[VAL_5]] {in_bounds = [true, true]} : tensor<80x16xf32>, vector<1x4xf32>
// CHECK:           %[[VAL_12:.*]] = vector.transfer_write %[[VAL_11]], %[[VAL_2]]{{\[}}%[[VAL_6]], %[[VAL_6]]] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<1x4xf32>
// CHECK:           return %[[VAL_12]] : tensor<1x4xf32>
// CHECK:         }

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
     %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_nd_extract } : (!transform.any_op) -> !transform.any_op
     transform.yield
   }
}

// -----

func.func @vectorize_nd_tensor_extract_with_tensor_extract(%input_1: tensor<1x20xi32>, %input_2: tensor<257x24xf32>, %arg0 : index, %arg1 : index, %arg2 : index, %arg3 : index) -> tensor<1x1x4xf32> {
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %output = tensor.empty() : tensor<1x1x4xf32>
  %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} outs(%output : tensor<1x1x4xf32>) {
  ^bb0(%out: f32):
    %13 = linalg.index 0 : index
    %14 = affine.apply affine_map<(d0, d1, d2) -> (d0 + d1 + d2)>(%arg0, %13, %arg2)
    %15 = linalg.index 2 : index
    %16 = linalg.index 1 : index
    %17 = affine.apply affine_map<(d0, d1, d2, d3) -> (d0 + d1 * 24 + d2 + d3)>(%arg1, %16, %15, %arg3)
    %extracted_0 = tensor.extract %input_1[%c0, %14] : tensor<1x20xi32>
    %18 = arith.index_cast %extracted_0 : i32 to index
    %19 = arith.maxsi %18, %c0 : index
    %20 = arith.minsi %19, %c256 : index
    %extracted_1 = tensor.extract %input_2[%20, %17] : tensor<257x24xf32>
    linalg.yield %extracted_1 : f32
  } -> tensor<1x1x4xf32>
  return %1 : tensor<1x1x4xf32>
}

// CHECK-LABEL:   func.func @vectorize_nd_tensor_extract_with_tensor_extract(
// CHECK-SAME:        %[[ARG0:.*]]: tensor<1x20xi32>, %[[ARG1:.*]]: tensor<257x24xf32>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index, %[[ARG5:.*]]: index) -> tensor<1x1x4xf32> {
//      CHECK:    %[[CST:.*]] = arith.constant dense<0> : vector<1x1x4xindex>
//      CHECK:    %[[CST0:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
//      CHECK:    %[[CST1:.*]] = arith.constant dense<256> : vector<1x1x4xindex>
//      CHECK:    %[[CST2:.*]] = arith.constant 0.000000e+00 : f32
//      CHECK:    %[[C0:.*]] = arith.constant 0 : index
//      CHECK:    %[[VAL0:.*]] = tensor.empty() : tensor<1x1x4xf32>
//      CHECK:    %[[VAL1:.*]] = arith.addi %[[ARG2]], %[[ARG4]] : index
//      CHECK:    %[[VAL2:.*]] = vector.broadcast %[[ARG3]] : index to vector<1x1x4xindex>
//      CHECK:    %[[VAL3:.*]] = vector.broadcast %[[CST0]] : vector<4xindex> to vector<1x1x4xindex>
//      CHECK:    %[[VAL4:.*]] = arith.addi %[[VAL2]], %[[VAL3]] : vector<1x1x4xindex>
//      CHECK:    %[[VAL5:.*]] = vector.broadcast %[[ARG5]] : index to vector<1x1x4xindex>
//      CHECK:    %[[VAL6:.*]] = arith.addi %[[VAL4]], %[[VAL5]] : vector<1x1x4xindex>
//      CHECK:    %[[EXTRACTED:.*]] = tensor.extract %[[ARG0]][%[[C0]], %[[VAL1]]] : tensor<1x20xi32>
//      CHECK:    %[[VAL7:.*]] = arith.index_cast %[[EXTRACTED]] : i32 to index
//      CHECK:    %[[VAL8:.*]] = vector.broadcast %[[VAL7]] : index to vector<1x1x4xindex>
//      CHECK:    %[[VAL9:.*]] = arith.maxsi %[[VAL8]], %[[CST]] : vector<1x1x4xindex>
//      CHECK:    %[[VAL10:.*]] = arith.minsi %[[VAL9]], %[[CST1]] : vector<1x1x4xindex>
//      CHECK:    %[[VAL11:.*]] = vector.extract %[[VAL10]][0, 0, 0] : index from vector<1x1x4xindex>
//      CHECK:    %[[VAL12:.*]] = vector.extract %[[VAL6]][0, 0, 0] : index from vector<1x1x4xindex>
//      CHECK:    %[[VAL13:.*]] = vector.transfer_read %[[ARG1]][%[[VAL11]], %[[VAL12]]], %[[CST2]] {in_bounds = [true, true]} : tensor<257x24xf32>, vector<1x4xf32>
//      CHECK:    %[[VAL14:.*]] = vector.broadcast %[[VAL13]] : vector<1x4xf32> to vector<1x1x4xf32>
//      CHECK:    %[[VAL15:.*]] = vector.transfer_write %[[VAL14]], %[[VAL0]][%[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true, true, true]} : vector<1x1x4xf32>, tensor<1x1x4xf32>
//      CHECK:    return %[[VAL15]] : tensor<1x1x4xf32>
//      CHECK:  }

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
     %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_nd_extract } : (!transform.any_op) -> !transform.any_op
     transform.yield
   }
}

// -----

// The vectorizer converts `affine.apply` so that the subsequent Ops can be vectorised based on the converted ops. Gather load.
func.func @vectorize_nd_tensor_extract_with_affine_apply_gather(%6: tensor<80x16xf32>, %arg0: index, %extracted_slice : tensor<1x4xf32>) -> tensor<1x4xf32> {
  %c16 = arith.constant 16 : index
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } outs(%extracted_slice : tensor<1x4xf32>) {
  ^bb0(%out: f32):
    %2 = linalg.index 1 : index
    %3 = affine.apply affine_map<(d0, d1) -> (d0 + d1)>(%2, %arg0)
    %extracted = tensor.extract %6[%3, %c16] : tensor<80x16xf32>
    linalg.yield %extracted : f32
  } -> tensor<1x4xf32>
  return %1 : tensor<1x4xf32>
}

// CHECK-LABEL:   func.func @vectorize_nd_tensor_extract_with_affine_apply_gather(
// CHECK-SAME:                                                                    %[[VAL_0:.*]]: tensor<80x16xf32>,
// CHECK-SAME:                                                                    %[[VAL_1:.*]]: index,
// CHECK-SAME:                                                                    %[[VAL_2:.*]]: tensor<1x4xf32>) -> tensor<1x4xf32> {
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<true> : vector<1x4xi1>
// CHECK:           %[[VAL_5:.*]] = arith.constant dense<0.000000e+00> : vector<1x4xf32>
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant dense<16> : vector<1x4xindex>
// CHECK:           %[[VAL_8:.*]] = vector.broadcast %[[VAL_1]] : index to vector<4xindex>
// CHECK:           %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_3]] : vector<4xindex>
// CHECK:           %[[VAL_10:.*]] = vector.broadcast %[[VAL_9]] : vector<4xindex> to vector<1x4xindex>
// CHECK:           %[[VAL_11:.*]] = arith.muli %[[VAL_10]], %[[VAL_7]] : vector<1x4xindex>
// CHECK:           %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_7]] : vector<1x4xindex>
// CHECK:           %[[VAL_13:.*]] = vector.gather %[[VAL_0]]{{\[}}%[[VAL_6]], %[[VAL_6]]] {{\[}}%[[VAL_12]]], %[[VAL_4]], %[[VAL_5]] : tensor<80x16xf32>, vector<1x4xindex>, vector<1x4xi1>, vector<1x4xf32> into vector<1x4xf32>
// CHECK:           %[[VAL_14:.*]] = vector.transfer_write %[[VAL_13]], %[[VAL_2]]{{\[}}%[[VAL_6]], %[[VAL_6]]] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<1x4xf32>
// CHECK:           return %[[VAL_14]] : tensor<1x4xf32>
// CHECK:         }

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
     %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_nd_extract } : (!transform.any_op) -> !transform.any_op
     transform.yield
   }
}

// -----

// Make sure that non-linear arithmetic operations (e.g. arith.maxsi) are allowed when calculating indices for load operations. Gather load.
func.func @vectorize_nd_tensor_extract_with_maxsi_gather(%arg0: tensor<80x16xf32>, %extracted_slice : tensor<1x4xf32>) -> tensor<1x4xf32> {
  %c79 = arith.constant 79 : index
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } outs(%extracted_slice : tensor<1x4xf32>) {
  ^bb0(%out: f32):
    %2 = linalg.index 1 : index
    %3 = arith.maxsi %2, %c79 : index
    %extracted = tensor.extract %arg0[%3, %2] : tensor<80x16xf32>
    linalg.yield %extracted : f32
  } -> tensor<1x4xf32>
  return %1 : tensor<1x4xf32>
}

// CHECK-LABEL:   func.func @vectorize_nd_tensor_extract_with_maxsi_gather(
// CHECK-SAME:                                                             %[[VAL_0:.*]]: tensor<80x16xf32>,
// CHECK-SAME:                                                             %[[VAL_1:.*]]: tensor<1x4xf32>) -> tensor<1x4xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<1264> : vector<1x4xindex>
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<true> : vector<1x4xi1>
// CHECK:           %[[VAL_5:.*]] = arith.constant dense<0.000000e+00> : vector<1x4xf32>
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = vector.broadcast %[[VAL_2]] : vector<4xindex> to vector<1x4xindex>
// CHECK:           %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %[[VAL_3]] : vector<1x4xindex>
// CHECK:           %[[VAL_9:.*]] = vector.gather %[[VAL_0]]{{\[}}%[[VAL_6]], %[[VAL_6]]] {{\[}}%[[VAL_8]]], %[[VAL_4]], %[[VAL_5]] : tensor<80x16xf32>, vector<1x4xindex>, vector<1x4xi1>, vector<1x4xf32> into vector<1x4xf32>
// CHECK:           %[[VAL_10:.*]] = vector.transfer_write %[[VAL_9]], %[[VAL_1]]{{\[}}%[[VAL_6]], %[[VAL_6]]] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<1x4xf32>
// CHECK:           return %[[VAL_10]] : tensor<1x4xf32>
// CHECK:         }

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
     %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_nd_extract } : (!transform.any_op) -> !transform.any_op
     transform.yield
   }
}

// -----

// Make sure that non-linear arithmetic operations (e.g. arith.maxsi) are allowed when calculating indices for load operations. Contiguous load.
func.func @vectorize_nd_tensor_extract_with_maxsi_contiguous(%arg0: tensor<80x16xf32>, %extracted_slice : tensor<1x4xf32>) -> tensor<1x4xf32> {
  %c16 = arith.constant 16 : index
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]
  } outs(%extracted_slice : tensor<1x4xf32>) {
  ^bb0(%out: f32):
    %2 = linalg.index 0 : index
    %3 = linalg.index 1 : index
    %4 = arith.maxsi %2, %c16 : index
    %extracted = tensor.extract %arg0[%4, %3] : tensor<80x16xf32>
    linalg.yield %extracted : f32
  } -> tensor<1x4xf32>
  return %1 : tensor<1x4xf32>
}

// CHECK-LABEL:   func.func @vectorize_nd_tensor_extract_with_maxsi_contiguous(
// CHECK-SAME:                                                                 %[[VAL_0:.*]]: tensor<80x16xf32>,
// CHECK-SAME:                                                                 %[[VAL_1:.*]]: tensor<1x4xf32>) -> tensor<1x4xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<16> : vector<4x1xindex>
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
// CHECK:           %[[VAL_5:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = vector.extract %[[VAL_2]][0, 0] : index from vector<4x1xindex>
// CHECK:           %[[VAL_6:.*]] = vector.extract %[[VAL_3]][0] : index from vector<4xindex>
// CHECK:           %[[VAL_8:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_7]], %[[VAL_6]]], %[[VAL_5]] {in_bounds = [true, true]} : tensor<80x16xf32>, vector<1x4xf32>
// CHECK:           %[[VAL_9:.*]] = vector.transfer_write %[[VAL_8]], %[[VAL_1]]{{\[}}%[[VAL_4]], %[[VAL_4]]] {in_bounds = [true, true]} : vector<1x4xf32>, tensor<1x4xf32>
// CHECK:           return %[[VAL_9]] : tensor<1x4xf32>
// CHECK:         }

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
     %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_nd_extract } : (!transform.any_op) -> !transform.any_op
     transform.yield
   }
}

// -----

// The vectorizer assumes it's a gather load whenever using a block argument to calculate an index.
#map = affine_map<(d0) -> (d0)>
func.func @vectorize_nd_tensor_extract_block_arg(%arg0: tensor<5x6xf32>, %arg1: tensor<5xindex>) -> tensor<5xf32> {
 %0 = tensor.empty() : tensor<5xf32>
 %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%arg1: tensor<5xindex>) outs(%0 : tensor<5xf32>) {
 ^bb0(%in: index, %out: f32):
   %2 = linalg.index 0 : index
   %extracted_0 = tensor.extract %arg0[%in, %2] : tensor<5x6xf32>
   linalg.yield %extracted_0 : f32
 } -> tensor<5xf32>
 return %1 : tensor<5xf32>
}

// CHECK-LABEL:   func.func @vectorize_nd_tensor_extract_block_arg(
// CHECK-SAME:                                                     %[[VAL_0:.*]]: tensor<5x6xf32>,
// CHECK-SAME:                                                     %[[VAL_1:.*]]: tensor<5xindex>) -> tensor<5xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<[0, 1, 2, 3, 4]> : vector<5xindex>
// CHECK:           %[[VAL_4:.*]] = arith.constant dense<true> : vector<5xi1>
// CHECK:           %[[VAL_5:.*]] = arith.constant dense<0.000000e+00> : vector<5xf32>
// CHECK:           %[[VAL_6:.*]] = arith.constant dense<6> : vector<5xindex>
// CHECK:           %[[VAL_7:.*]] = tensor.empty() : tensor<5xf32>
// CHECK:           %[[VAL_8:.*]] = vector.transfer_read %[[VAL_1]]{{\[}}%[[VAL_2]]], %[[VAL_2]] {in_bounds = [true]} : tensor<5xindex>, vector<5xindex>
// CHECK:           %[[VAL_9:.*]] = arith.muli %[[VAL_8]], %[[VAL_6]] : vector<5xindex>
// CHECK:           %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_3]] : vector<5xindex>
// CHECK:           %[[VAL_11:.*]] = vector.gather %[[VAL_0]]{{\[}}%[[VAL_2]], %[[VAL_2]]] {{\[}}%[[VAL_10]]], %[[VAL_4]], %[[VAL_5]] : tensor<5x6xf32>, vector<5xindex>, vector<5xi1>, vector<5xf32> into vector<5xf32>
// CHECK:           %[[VAL_12:.*]] = vector.transfer_write %[[VAL_11]], %[[VAL_7]]{{\[}}%[[VAL_2]]] {in_bounds = [true]} : vector<5xf32>, tensor<5xf32>
// CHECK:           return %[[VAL_12]] : tensor<5xf32>
// CHECK:         }

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
     %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_nd_extract } : (!transform.any_op) -> !transform.any_op
     transform.yield
   }
}

// -----

#map1 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
func.func @vectorize_0d_tensor_extract(%arg0: tensor<f32>, %arg2: tensor<1x1x3xf32>) -> tensor<1x1x3xf32> {
  %2 = linalg.generic {
    indexing_maps = [#map1],
    iterator_types = ["parallel", "parallel", "parallel"]
  } outs(%arg2 : tensor<1x1x3xf32>) {
  ^bb0(%arg4: f32):
    %7 = tensor.extract %arg0[] : tensor<f32>
    linalg.yield %7 : f32
  } -> tensor<1x1x3xf32>
  return %2 : tensor<1x1x3xf32>
}

// CHECK-LABEL:   func.func @vectorize_0d_tensor_extract(
// CHECK-SAME:     %[[ARG_0:.*]]: tensor<f32>
// CHECK:           %[[EXTRACT:.*]] = tensor.extract %[[ARG_0]][] : tensor<f32>
// CHECK:           vector.broadcast %[[EXTRACT]] : f32 to vector<1x1x3xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
     %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_nd_extract } : (!transform.any_op) -> !transform.any_op
     transform.yield
   }
}

// -----

func.func @vectorize_nd_tensor_extract_contigous(%arg0: tensor<80x16x17x18x19xf32>, %extracted_slice : tensor<4x5x6x7x8xf32>) -> tensor<4x5x6x7x8xf32> {
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]
  } outs(%extracted_slice : tensor<4x5x6x7x8xf32>) {
  ^bb0(%out: f32):
    %2 = linalg.index 0 : index
    %3 = linalg.index 1 : index
    %4 = linalg.index 2 : index
    %5 = linalg.index 3 : index
    %6 = linalg.index 4 : index
    %extracted = tensor.extract %arg0[%2, %3, %4, %5, %6] : tensor<80x16x17x18x19xf32>
    linalg.yield %extracted : f32
  } -> tensor<4x5x6x7x8xf32>
  return %1 : tensor<4x5x6x7x8xf32>
}

// CHECK-LABEL:   func.func @vectorize_nd_tensor_extract_contigous(
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<80x16x17x18x19xf32>, %[[ARG1:.*]]: tensor<4x5x6x7x8xf32>) -> tensor<4x5x6x7x8xf32> {
//       CHECK:    %[[CST:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
//       CHECK:    %[[CST0:.*]] = arith.constant dense<[0, 1, 2, 3, 4]> : vector<5xindex>
//       CHECK:    %[[CST1:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5]> : vector<6xindex>
//       CHECK:    %[[CST2:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6]> : vector<7xindex>
//       CHECK:    %[[CST3:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7]> : vector<8xindex>
//       CHECK:    %[[CST4:.*]] = arith.constant 0.000000e+00 : f32
//       CHECK:    %[[C0:.*]] = arith.constant 0 : index
//       CHECK:    %[[VAL0:.*]] = vector.extract %[[CST]][0] : index from vector<4xindex>
//       CHECK:    %[[VAL1:.*]] = vector.extract %[[CST0]][0] : index from vector<5xindex>
//       CHECK:    %[[VAL2:.*]] = vector.extract %[[CST1]][0] : index from vector<6xindex>
//       CHECK:    %[[VAL3:.*]] = vector.extract %[[CST2]][0] : index from vector<7xindex>
//       CHECK:    %[[VAL4:.*]] = vector.extract %[[CST3]][0] : index from vector<8xindex>
//       CHECK:    %[[VAL5:.*]] = vector.transfer_read %[[ARG0]][%[[VAL0]], %[[VAL1]], %[[VAL2]], %[[VAL3]], %[[VAL4]]], %[[CST4]] {in_bounds = [true, true, true, true, true]} : tensor<80x16x17x18x19xf32>, vector<4x5x6x7x8xf32>
//       CHECK:    %[[VAL6:.*]] = vector.transfer_write %[[VAL5]], %arg1[%[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true, true, true, true, true]} : vector<4x5x6x7x8xf32>, tensor<4x5x6x7x8xf32>
//       CHECK:    return %[[VAL6]] : tensor<4x5x6x7x8xf32>
//       CHECK:  }

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
     %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_nd_extract } : (!transform.any_op) -> !transform.any_op
     transform.yield
   }
}

// -----

func.func @vectorize_nd_tensor_extract_gather(%arg0: tensor<80x16x17x18x19xf32>, %extracted_slice : tensor<4x5x6x7xf32>) -> tensor<4x5x6x7xf32> {
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } outs(%extracted_slice : tensor<4x5x6x7xf32>) {
  ^bb0(%out: f32):
    %2 = linalg.index 0 : index
    %3 = linalg.index 1 : index
    %4 = linalg.index 2 : index
    %5 = linalg.index 3 : index
    %extracted = tensor.extract %arg0[%2, %3, %4, %5, %5] : tensor<80x16x17x18x19xf32>
    linalg.yield %extracted : f32
  } -> tensor<4x5x6x7xf32>
  return %1 : tensor<4x5x6x7xf32>
}

// CHECK-LABEL:   func.func @vectorize_nd_tensor_extract_gather(
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<80x16x17x18x19xf32>, %[[ARG1:.*]]: tensor<4x5x6x7xf32>) -> tensor<4x5x6x7xf32> {
//       CHECK:   %[[CST:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[VAL:.*]] = vector.gather %[[ARG0]][%[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]]] [%{{.*}}], %{{.*}}, %{{.*}} : tensor<80x16x17x18x19xf32>, vector<4x5x6x7xindex>, vector<4x5x6x7xi1>, vector<4x5x6x7xf32> into vector<4x5x6x7xf32>
//       CHECK:   %{{.*}} = vector.transfer_write %[[VAL]], %[[ARG1]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true, true, true, true]} : vector<4x5x6x7xf32>, tensor<4x5x6x7xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
     %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_nd_extract } : (!transform.any_op) -> !transform.any_op
     transform.yield
   }
}

// -----

func.func @vectorize_nd_tensor_extract_gather_constant_indices(%arg0: tensor<80x16x17x18x19xf32>, %extracted_slice : tensor<6x7x8xf32>) -> tensor<6x7x8xf32> {
  %c5 = arith.constant 5 : index
  %1 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } outs(%extracted_slice : tensor<6x7x8xf32>) {
  ^bb0(%out: f32):
    %2 = linalg.index 0 : index
    %3 = linalg.index 1 : index
    %4 = linalg.index 2 : index
    %extracted = tensor.extract %arg0[%c5, %c5, %2, %3, %4] : tensor<80x16x17x18x19xf32>
    linalg.yield %extracted : f32
  } -> tensor<6x7x8xf32>
  return %1 : tensor<6x7x8xf32>
}

// CHECK-LABEL:   func.func @vectorize_nd_tensor_extract_gather_constant_indices(
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<80x16x17x18x19xf32>, %[[ARG1:.*]]: tensor<6x7x8xf32>) -> tensor<6x7x8xf32> {
//       CHECK:   %[[CST:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5]> : vector<6xindex>
//       CHECK:   %[[C0:.*]] = arith.constant 0 : index
//       CHECK:   %[[VAL:.*]] = vector.gather %[[ARG0]][%[[C0]], %[[C0]], %[[C0]], %[[C0]], %[[C0]]] [%{{.*}}], %{{.*}}, %{{.*}} : tensor<80x16x17x18x19xf32>, vector<6x7x8xindex>, vector<6x7x8xi1>, vector<6x7x8xf32> into vector<6x7x8xf32>
//       CHECK:   %{{.*}} = vector.transfer_write %[[VAL]], %[[ARG1]][%[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true, true, true]} : vector<6x7x8xf32>, tensor<6x7x8xf32>

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
     %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_nd_extract } : (!transform.any_op) -> !transform.any_op
     transform.yield
   }
}

// -----

func.func @vectorize_nd_tensor_extract_contigous_complex(%6: tensor<45x80x16x17xf32>, %arg0: index, %arg1: index, %arg2: index, %arg3: index, %extracted_slice : tensor<1x4x5x6xf32>) -> tensor<1x4x5x6xf32> {
  %0 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } outs(%extracted_slice : tensor<1x4x5x6xf32>) {
  ^bb0(%out: f32):
    %1 = linalg.index 0 : index
    %2 = linalg.index 1 : index
    %3 = linalg.index 2 : index
    %4 = linalg.index 3 : index
    
    %21 = arith.addi %arg0, %1 : index
    %22 = arith.addi %21, %arg1 : index
    
    %23 = arith.addi %arg0, %2 : index
    %24 = arith.addi %23, %arg2 : index

    %25 = arith.addi %arg1, %3 : index
    %26 = arith.addi %arg3, %25 : index

    %27 = arith.addi %arg2, %4 : index
    %28 = arith.addi %arg3, %27 : index

    %extracted = tensor.extract %6[%22, %24, %26, %28] : tensor<45x80x16x17xf32>
    linalg.yield %extracted : f32
  } -> tensor<1x4x5x6xf32>
  return %0 : tensor<1x4x5x6xf32>
}

// CHECK-LABEL:   func.func @vectorize_nd_tensor_extract_contigous_complex(
//  CHECK-SAME:     %[[ARG0:.*]]: tensor<45x80x16x17xf32>, %[[ARG1:.*]]: index, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index, %[[ARG5:.*]]: tensor<1x4x5x6xf32>) -> tensor<1x4x5x6xf32> {
//       CHECK:       %[[CST:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xindex>
//       CHECK:       %[[CST0:.*]] = arith.constant dense<[0, 1, 2, 3, 4]> : vector<5xindex>
//       CHECK:       %[[CST1:.*]] = arith.constant dense<[0, 1, 2, 3, 4, 5]> : vector<6xindex>
//       CHECK:       %[[CST2:.*]] = arith.constant 0.000000e+00 : f32
//       CHECK:       %[[C0:.*]] = arith.constant 0 : index
//       CHECK:       %[[VAL0:.*]] = vector.broadcast %[[CST]] : vector<4xindex> to vector<1x6x5x4xindex>
//       CHECK:       %[[VAL1:.*]] = vector.transpose %[[VAL0]], [0, 3, 2, 1] : vector<1x6x5x4xindex> to vector<1x4x5x6xindex>
//       CHECK:       %[[VAL2:.*]] = vector.broadcast %[[CST0]] : vector<5xindex> to vector<1x4x6x5xindex>
//       CHECK:       %[[VAL3:.*]] = vector.transpose %[[VAL2]], [0, 1, 3, 2] : vector<1x4x6x5xindex> to vector<1x4x5x6xindex>
//       CHECK:       %[[VAL4:.*]] = arith.addi %[[ARG1]], %[[ARG2]] : index
//       CHECK:       %[[VAL5:.*]] = vector.broadcast %[[ARG1]] : index to vector<1x4x5x6xindex>
//       CHECK:       %[[VAL6:.*]] = arith.addi %[[VAL5]], %[[VAL1]] : vector<1x4x5x6xindex>
//       CHECK:       %[[VAL7:.*]] = vector.broadcast %[[ARG3]] : index to vector<1x4x5x6xindex>
//       CHECK:       %[[VAL8:.*]] = arith.addi %[[VAL6]], %[[VAL7]] : vector<1x4x5x6xindex>
//       CHECK:       %[[VAL9:.*]] = vector.broadcast %[[ARG2]] : index to vector<1x4x5x6xindex>
//       CHECK:       %[[VAL10:.*]] = arith.addi %[[VAL9]], %[[VAL3]] : vector<1x4x5x6xindex>
//       CHECK:       %[[VAL11:.*]] = vector.broadcast %[[ARG4]] : index to vector<1x4x5x6xindex>
//       CHECK:       %[[VAL12:.*]] = arith.addi %[[VAL11]], %[[VAL10]] : vector<1x4x5x6xindex>
//       CHECK:       %[[VAL13:.*]] = vector.broadcast %[[ARG3]] : index to vector<6xindex>
//       CHECK:       %[[VAL14:.*]] = arith.addi %[[VAL13]], %[[CST1]] : vector<6xindex>
//       CHECK:       %[[VAL15:.*]] = vector.broadcast %[[ARG4]] : index to vector<6xindex>
//       CHECK:       %[[VAL16:.*]] = arith.addi %[[VAL15]], %[[VAL14]] : vector<6xindex>
//       CHECK:       %[[VAL17:.*]] = vector.extract %[[VAL8]][0, 0, 0, 0] : index from vector<1x4x5x6xindex>
//       CHECK:       %[[VAL18:.*]] = vector.extract %[[VAL12]][0, 0, 0, 0] : index from vector<1x4x5x6xindex>
//       CHECK:       %[[VAL19:.*]] = vector.extract %[[VAL16]][0] : index from vector<6xindex>
//       CHECK:       %[[VAL20:.*]] = vector.transfer_read %[[ARG0]][%[[VAL4]], %[[VAL17]], %[[VAL18]], %[[VAL19]]], %[[CST2]] {in_bounds = [true, true, true, true]} : tensor<45x80x16x17xf32>, vector<1x4x5x6xf32>
//       CHECK:       %[[VAL21:.*]] = vector.transfer_write %[[VAL20]], %[[ARG5]][%[[C0]], %[[C0]], %[[C0]], %[[C0]]] {in_bounds = [true, true, true, true]} : vector<1x4x5x6xf32>, tensor<1x4x5x6xf32>
//       CHECK:       return %[[VAL21]] : tensor<1x4x5x6xf32>
//       CHECK:     }

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
     %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
     %1 = transform.get_parent_op %0 {isolated_from_above} : (!transform.any_op) -> !transform.any_op
     %2 = transform.structured.vectorize_children_and_apply_patterns %1 { vectorize_nd_extract } : (!transform.any_op) -> !transform.any_op
     transform.yield
   }
}
